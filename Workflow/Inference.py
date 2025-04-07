import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '.')
import importlib
import h5py
import io
import numpy as np
from tqdm import tqdm
import networks.Models as ms
from data_loader.data_loader_2 import H5DataLoader
import common.user as user
import common.data_structure as data_structure
from common.selections import selections
from functools import reduce
import pickle
import copy
import pandas as pd

import logging
logger = logging.getLogger('UNC')

# A functor to make MVA based selections for Poisson regions
def makeMVAselector( class_name, lower, upper, selection_mva): 
    def _selector( data, class_name=class_name, lower=lower, upper=upper, selection_mva=selection_mva):
        pred = selection_mva.predict(data[:, :28], ic_scaling=False)[:,data_structure.labels.index(class_name)]
        mask = (pred >= lower ) & (pred < upper)
        return data[mask]
    return _selector 

class Inference:
    def __init__(self, cfg, small=False, overwrite=False, toy_origin="config", toy_path=None, toy_from_memory=None):
        """
        Initialize the Inference object.
        Load configuration, models, and parameters.

        Args:
            cfg_path (str): Path to the configuration file.
            small (bool): Whether to use a smaller dataset.
        """

        self.cfg = cfg

        # Ensure required sections are defined in the config
        assert "Tasks" in self.cfg, "Section Tasks not defined in config!"
        assert "Selections" in self.cfg, "Section Selections not defined in config!"

        # Require toy origin to be defined
        assert toy_origin in ["config", "path", "memory"], "toy_origin is not well defined!"
        logger.info("Load toy from {}.".format(toy_origin))

        # Initialize attributes
        self.training_data = {}
        self.models        = {}
        self.teststat      = {}
        self.selections    = self.cfg['Selections']
        self.small         = small
        self.overwrite     = overwrite
        self.h5s           = {}
        self.toy_origin    = toy_origin
        self.toy_path      = toy_path
        self.toy_from_memory = toy_from_memory

        self.ignore_check = False

        # Delete toy from config if not needed
        if toy_origin != "config":
            if "Toy" in self.cfg["Save"]:
                del self.cfg["Save"]["Toy"]
                logger.info("Specified toy from path or memory, remove toy from config")

        # Load models and parameters
        self.load_models()
        self.load_icps()

        self.alpha_bkg     = self.cfg['Parameters']['alpha_bkg']
        self.alpha_tt      = self.cfg['Parameters']['alpha_tt']
        self.alpha_diboson = self.cfg['Parameters']['alpha_diboson']

        self.float_parameters = {
            "nu_tes":False,
            "nu_jes":False,
            "nu_met":False,
            "nu_bkg":False,
            "nu_tt":False,
            "nu_diboson":False,
        }

        # ----------------------------------------------------------
        # Determine whether we do CSIs
        # ----------------------------------------------------------
        if self.cfg.get('CSI') is not None:
            try:
                self.load_csis()
                logger.info("Loaded existing CSIs.")
            except (IOError, AssertionError, TypeError):
                logger.info("Error loading CSIs. Will proceed.")
                pass

        self.load_calibrations()
        self.load_Poisson_predictions()

    def ignore_loading_check(self):
        """
        ignores the checks on whether the paths of ML models are consistent
        when loading H5 files
        """
        self.ignore_check=True

    def setToyFromMemory(self, toy):
        self.toy_origin    = "memory"
        self.toy_path      = None
        self.toy_from_memory = toy
        # Delete toy from config if not needed
        if self.toy_origin != "config":
            if "Toy" in self.cfg["Save"]:
                del self.cfg["Save"]["Toy"]
                logger.info("Specified toy from path or memory, remove toy from config")
        self.loadToyFromMemory(ignore_done=True) # load and force overwrite


    def calibrate_dcr(self, selection, input_dcr):
        """
        calibrates the input (in DCR space) based on the loaded calibrator
        """
        if selection not in self.calibrations:
            return input_dcr
        else:
            output_dcr = self.calibrations[selection].predict(input_dcr)

        return output_dcr

    def training_data_loader(self, selection, n_split):
        """
        Load training data for a given selection.

        Args:
            selection (str): Selection criteria.
            n_split (int): Number of splits for the dataset.

        Returns:
            DataLoader: The loaded training data.
        """
        import common.datasets_hephy as datasets_hephy
        d = datasets_hephy.get_data_loader(selection=selection, n_split=n_split)
        logger.info("Training data loaded for selection: {}".format(selection))
        return d

    def load_toy_file(self, filename, batch_size, n_split, selection_function=None):
        """
        Load a toy dataset from an HDF5 file.

        Args:
            filename (str): Path to the toy file.
            batch_size (int): Batch size for loading.
            n_split (int): Number of splits for the dataset.

        Returns:
            H5DataLoader: The loaded toy dataset.
        """
        assert os.path.exists(filename), "Toy file {} does not exist!".format(filename)
        t = H5DataLoader(
            file_path          = filename,
            batch_size         = batch_size,
            n_split            = n_split,
            selection_function = selection_function,
        )
        logger.info("Toy loaded from {}.".format(filename))
        return t

    def loadH5(self, filename, selection):
        """
        Load an HDF5 file and validate its consistency with the configuration.

        Args:
            filename (str): Path to the HDF5 file.
            selection (str): Selection criteria.

        Returns:
            h5py.File: The loaded HDF5 file.
        """
        try:
            h5f = h5py.File(filename)
        except BlockingIOError as e:
            logger.error(f"File {filename} blocked.")
            raise e

        # Validate the model path and module consistency
        if not self.ignore_check:
            for t in self.cfg['Tasks']:
                assert h5f.attrs[t + "_module"] == self.cfg[t][selection]['module'], \
                    "Task {} selection {}: inconsistent module! H5: {} -- Config: {}".format(
                        t, selection, h5f.attrs[t + "_module"], self.cfg[t][selection]['module'])
                assert h5f.attrs[t + "_model_path"] == self.cfg[t][selection]['model_path'], \
                    "Task {} selection {}: inconsistent model path! H5 {} -- Config {}".format(
                        t, selection, h5f.attrs[t + "_model_path"], self.cfg[t][selection]['model_path'])
        return h5f

    def load_icps( self ):
        logger.info("Loading ICPs.")
        self.icps={}
        counter = {}
        for t in self.cfg['Tasks']:
            if t=='MultiClassifier': continue
            counter[t]=0
            for s in self.cfg["Selections"]:
                if s in self.cfg[t] and 'icp_file' in self.cfg[t][s]:
                    if t not in self.icps:
                        self.icps[t] = {}
                    self.icps[t][s] = ms.InclusiveCrosssectionParametrization.load(self.cfg[t][s]['icp_file'])
                    counter[t]+=1

        logger.info( "Loaded ICPs: "+" ".join( ["%s: %i"%(t,counter[t]) for t in self.cfg['Tasks'] if t!='MultiClassifier'] ) )

    def loadMLresults(self, name, filename, ignore_done=False):
        """
        Load machine learning results from an HDF5 file.

        Args:
            name (str): Name of the results.
            filename (str): Base name of the HDF5 file.
            ignore_done (bool): Whether to ignore already loaded results.
        """
        for selection in self.selections:
            if not ignore_done and name in self.h5s and selection in self.h5s[name]:
                continue  # Results already loaded, skip

            logger.debug( f"Loading ML results for {name} in selection {selection}" )
            h5_filename = os.path.join( self.cfg['tmp_path'], filename + '_' + selection + '.h5' )
            assert os.path.exists(h5_filename), "File {} does not exist! Try running the save mode first.".format(h5_filename)

            h5f = self.loadH5(h5_filename, selection)
            if name not in self.h5s:
                self.h5s[name] = {}

            # Load datasets from HDF5
            self.h5s[name][selection] = {
                "MultiClassifier_predict": h5f["MultiClassifier_predict"][:],
                "htautau_DeltaA":          h5f["htautau_DeltaA"][:],
                "ztautau_DeltaA":          h5f["ztautau_DeltaA"][:],
                "ttbar_DeltaA":            h5f["ttbar_DeltaA"][:],
                "diboson_DeltaA":          h5f["diboson_DeltaA"][:],
                "Weight":                  h5f["Weight"][:],
                "Label":                   h5f["Label"][:]
            }
            logger.info("ML results {} with {} loaded from {}".format(name, selection, h5_filename))

        if name!="TrainingData" and "Poisson" in self.cfg:
            for name, poisson_data in self.poisson.items():
                # Check whether we have it already
                if poisson_data['observation'] is not None or ( 'ignore' in poisson_data and poisson_data['ignore']):
                    continue
                pkl_filename = os.path.join(
                    self.cfg['tmp_path'], f"Poisson_{name}_Toy.pkl")
                if not os.path.exists(pkl_filename):
                    raise RuntimeError(f"Poisson term file {pkl_filename} not found. Run with --save first.") 
                with open(pkl_filename, 'rb') as pkl_file:
                    poisson_data['observation']=pickle.load( pkl_file)
                    logger.info(f"Read Poisson observation from {pkl_filename}. Found {name}: {poisson_data['observation']}")

    def loadToyFromPath(self, filename, ignore_done=False):
        """
        Load toy directly from raw -h5 file
        We calculate the arrays with the ML ntuple informations on the fly,
        do not write an output file and store the arrays in the corresponding self.h5s["Toy"]
        Args:
            filename (str): Path of the toy.
            ignore_done (bool): Whether to ignore already loaded results.
        """

        assert os.path.exists(filename), "Toy file {} does not exist!".format(filename)
        for selection in self.selections:
            if not ignore_done and "Toy" in self.h5s and selection in self.h5s["Toy"]:
                return  # Results already loaded, skip

            # Load h5 file
            rawToy = self.load_toy_file(
                filename = filename,
                batch_size = None,
                n_split = 1,
                selection_function = selections[selection]
            )

            # Check if the dict already exists, otherwise create it
            if "Toy" not in self.h5s:
                self.h5s["Toy"] = {}
            if selection not in self.h5s["Toy"]:
                self.h5s["Toy"][selection] = {}

            # Load features, weight, labels and convert to ML scores/DeltaAs
            with tqdm(total=len(rawToy), desc="Processing batches") as pbar:
                for i_batch, batch in enumerate(rawToy):
                    features, weights, labels = rawToy.split(batch)
                    MLpredictions = {}
                    for t in self.cfg['Tasks']:
                        if "save" not in self.cfg[t]:
                            continue

                        for iobj in self.cfg[t]['save']:
                            ds_name = t + '_' + iobj

                            if iobj == "predict":
                                pred = self.models[t][selection].predict(features)
                            elif iobj == "DeltaA":
                                pred = self.models[t][selection].get_DeltaA(features)
                            else:
                                raise Exception(
                                    f"Unsupported save type: '{iobj}'. "
                                    "Currently supported: 'predict', 'DeltaA'."
                                )
                            MLpredictions[ds_name] = pred

                    # Save in dict
                    if i_batch == 0:
                        self.h5s["Toy"][selection]["MultiClassifier_predict"] = MLpredictions["MultiClassifier_predict"][:]
                        self.h5s["Toy"][selection]["htautau_DeltaA"] = MLpredictions["htautau_DeltaA"][:]
                        self.h5s["Toy"][selection]["ztautau_DeltaA"] = MLpredictions["ztautau_DeltaA"][:]
                        self.h5s["Toy"][selection]["ttbar_DeltaA"] = MLpredictions["ttbar_DeltaA"][:]
                        self.h5s["Toy"][selection]["diboson_DeltaA"] = MLpredictions["diboson_DeltaA"][:]
                        self.h5s["Toy"][selection]["Weight"] = weights[:]
                        self.h5s["Toy"][selection]["Label"] = labels[:]
                        self.h5s["Toy"][selection]["features"] = features[:] #A flaw in the code design. Inference loads its data. Now that I need the toy features I need to accumulate them here. The data should be handed over, not loaded here or anywhere in this class.
                    else:
                        self.h5s["Toy"][selection]["MultiClassifier_predict"] = np.append(self.h5s["Toy"][selection]["MultiClassifier_predict"], MLpredictions["MultiClassifier_predict"][:])
                        self.h5s["Toy"][selection]["htautau_DeltaA"] = np.append(self.h5s["Toy"][selection]["htautau_DeltaA"], MLpredictions["htautau_DeltaA"][:])
                        self.h5s["Toy"][selection]["ztautau_DeltaA"] = np.append(self.h5s["Toy"][selection]["ztautau_DeltaA"], MLpredictions["ztautau_DeltaA"][:])
                        self.h5s["Toy"][selection]["ttbar_DeltaA"] = np.append(self.h5s["Toy"][selection]["ttbar_DeltaA"], MLpredictions["ttbar_DeltaA"][:])
                        self.h5s["Toy"][selection]["diboson_DeltaA"] = np.append(self.h5s["Toy"][selection]["diboson_DeltaA"], MLpredictions["diboson_DeltaA"][:])
                        self.h5s["Toy"][selection]["Weight"] = np.append(self.h5s["Toy"][selection]["Weight"], weights[:])
                        self.h5s["Toy"][selection]["Label"] = np.append(self.h5s["Toy"][selection]["Label"], labels[:])
                        self.h5s["Toy"][selection]["features"] = np.append(self.h5s["Toy"][selection]["features"], features[:])
                    
                    pbar.update(1)

            logger.info("Toy with {} loaded from {}".format(selection, filename))

            # Poisson data
            if "Poisson" in self.cfg:
                for name, poisson_data in self.poisson.items():
                    if "ignore" in self.cfg['Poisson'][name] and self.cfg['Poisson'][name]["ignore"]: continue
                    poisson_rawToy = self.load_toy_file( filename = filename, batch_size = None, n_split = 1, selection_function = selections[self.cfg["Poisson"][name]["preselection"]])
                    poisson_data['observation'] = self.Poisson_observation( data_input = poisson_rawToy, selectors = poisson_data['mva_selectors'], small=False)
                    logger.info(f"loadToyFromPath: Computed Poisson observation {name}: {poisson_data['observation']}")

    def convertToyToDataStruct(self):
        """
        Convert toy dataset into HDF5 format and store it in memory before event selection.
        """
        assert self.toy_from_memory is not None, "Toy dataset not defined!"

        if isinstance(self.toy_from_memory["data"], pd.Series) or isinstance(self.toy_from_memory["data"], pd.DataFrame):
            features = self.toy_from_memory["data"].to_numpy()  # (N, 28)
        else:
            features = self.toy_from_memory["data"]  # (N, 28)

        if isinstance(self.toy_from_memory["weights"], pd.Series) or isinstance(self.toy_from_memory["weights"], pd.DataFrame):
            weights = self.toy_from_memory["weights"].to_numpy().reshape(-1, 1)  # (N, 1)
        else:
            weights = self.toy_from_memory["weights"].reshape(-1, 1)  # (N, 1)

        labels = np.full((features.shape[0], 1), -1)  # (N, 1), all -1

        # Convert into our format
        converted_data = np.hstack((features, weights, labels))

        return converted_data

    def loadToyFromMemory(self, ignore_done=False):
        assert self.toy_from_memory is not None, "Toy not defined!"

        for selection in self.selections:
            if not ignore_done and "Toy" in self.h5s and selection in self.h5s["Toy"]:
                return  # Results already loaded, skip

            # Check if the dict already exists, otherwise create it
            if "Toy" not in self.h5s:
                self.h5s["Toy"] = {}
            if selection not in self.h5s["Toy"]:
                self.h5s["Toy"][selection] = {}

            # convert toy in our data format and load data
            toy_data = self.convertToyToDataStruct()

            # Make event selection
            selection_function = selections[selection]
            selected_toy_data = selection_function(toy_data)

            # Split into features and weights:
            features = selected_toy_data[:, :len(data_structure.feature_names)]
            weights  = selected_toy_data[:, data_structure.weight_index]
            labels   = selected_toy_data[:, data_structure.label_index]

            # Get ML info from features
            MLpredictions = {}
            for t in self.cfg['Tasks']:
                if "save" not in self.cfg[t]:
                    continue

                for iobj in self.cfg[t]['save']:
                    ds_name = t + '_' + iobj

                    if iobj == "predict":
                        pred = self.models[t][selection].predict(features)
                    elif iobj == "DeltaA":
                        pred = self.models[t][selection].get_DeltaA(features)
                    else:
                        raise Exception(
                            f"Unsupported save type: '{iobj}'. "
                            "Currently supported: 'predict', 'DeltaA'."
                        )
                    MLpredictions[ds_name] = pred

            # Save in dict
            self.h5s["Toy"][selection]["MultiClassifier_predict"] = MLpredictions["MultiClassifier_predict"][:]
            self.h5s["Toy"][selection]["htautau_DeltaA"] = MLpredictions["htautau_DeltaA"][:]
            self.h5s["Toy"][selection]["ztautau_DeltaA"] = MLpredictions["ztautau_DeltaA"][:]
            self.h5s["Toy"][selection]["ttbar_DeltaA"] = MLpredictions["ttbar_DeltaA"][:]
            self.h5s["Toy"][selection]["diboson_DeltaA"] = MLpredictions["diboson_DeltaA"][:]
            self.h5s["Toy"][selection]["Weight"] = weights[:]
            self.h5s["Toy"][selection]["Label"] = labels[:]

            logger.info("Toy with {} loaded from memory".format(selection))


        # Poisson data
        if "Poisson" in self.cfg:
            for name, poisson_data in self.poisson.items():
                if "ignore" in self.cfg['Poisson'][name] and self.cfg['Poisson'][name]["ignore"]: continue
                # convert toy in our data format and load data
                if ignore_done or not (poisson_data['observation'] is not None):
                    toy_data = self.convertToyToDataStruct()

                    # Apply selections
                    selected_toy_data = reduce( lambda acc, f: f(acc), [poisson_data["preselector"]]+poisson_data['mva_selectors'], toy_data )

                    # Store the Poisson data per label, including -1
                    labels  = selected_toy_data[:, data_structure.label_index]
                    poisson_data['observation'] = {}
                    for i_label in [-1] + list(range(len(data_structure.labels))):
                        label_mask = (labels==i_label)
                        poisson_data['observation'][i_label] = selected_toy_data[label_mask,data_structure.weight_index].sum()
                    logger.info(f"loadToyFromMemory: Computed Poisson observation {name}: {poisson_data['observation']}")

    def load_models(self):
        """
        Load models for all tasks and selections defined in the configuration.
        """
        for t in self.cfg['Tasks']:
            assert t in self.cfg, "{t} not defined in config!"
            self.models[t] = {}

            for s in self.selections:
                assert s in self.cfg[t], "{s} not defined in {t} in the config!"

                m = ms.getModule(self.cfg[t][s]["module"])
                self.models[t][s] = m.load(self.cfg[t][s]["model_path"])
                logger.info("Task {} selection {}: Module {} loaded with model path {}.".format(
                    t, s, self.cfg[t][s]["module"], self.cfg[t][s]["model_path"]))

    def load_csis(self):
        """
        Load the csis (interpolators).

        """
        # Already loaded
        if hasattr( self, "csis" ): return
        self.csis = {}
        self.csis_const = {}
        for s in self.selections:
            self.csis[s] = {}
            self.csis_const[s] = {}
            for t in self.cfg['Tasks'][1:]:
                pkl_filename = os.path.join( self.cfg['tmp_path'], "CSI_%s_%s_TrainingData.pkl"%(s,t) )
                assert os.path.exists(pkl_filename), "CSIs file {} does not exist!".format(pkl_filename)
                with open(pkl_filename, 'rb') as f:
                    self.csis[s][t], self.csis_const[s][t] = pickle.load(f)

                logger.info(f"CSI loaded from {pkl_filename}.")

    def load_calibrations(self):
        """
        Load the calibrations.

        """

        # Already loaded
        if hasattr( self, "calibrations" ): return
        self.calibrations = {}
        for s in self.selections:
            if 'calibration' in self.cfg['MultiClassifier'][s]:
                pkl_filename = self.cfg['MultiClassifier'][s]['calibration']
                assert os.path.exists(pkl_filename), "calibrations file {} does not exist!".format(pkl_filename)

                calib_class = self.cfg['MultiClassifier'][s]["calibration_module"] if "calibration_module" in self.cfg['MultiClassifier'][s] else "Calibration"
                self.calibrations[s] = ms.getModule( calib_class ).load( pkl_filename ) 
                logger.info(f"Calibration {calib_class} loaded for {s} from {pkl_filename}.")

    def load_Poisson_predictions(self):
        self.poisson={}
        if "Poisson" not in self.cfg:
            return

        for name, poisson_cfg in self.cfg["Poisson"].items():
            if "ignore" in poisson_cfg and poisson_cfg["ignore"]: continue
            logger.info(f"Loading data for Poisson region: {name}")
            poisson = { 
                'preselector': selections[poisson_cfg['preselection']], # functor to apply the pre-selection
                'IC'         : ms.InclusiveCrosssection.load( poisson_cfg['IC'] ), # the inclusive cross sections per process in the selection identified by 'name'
                'ICP'        : { process:ms.InclusiveCrosssectionParametrization.load( poisson_cfg['ICP'][process]) for process in data_structure.labels }, # systematics dependence 
                'observation': None,    # The observation never changes. We need to compute it from the toy.
                } 
            self.poisson[name] = poisson
            poisson["mva_selectors"] = []
            if 'mva_selection' in poisson_cfg:
                m = ms.getModule(poisson_cfg["module"])
                poisson['selection_mva'] = m.load(poisson_cfg["model_path"])
                for class_name, lower, upper in poisson_cfg['mva_selection']: 
                    poisson["mva_selectors"].append( makeMVAselector( class_name, lower, upper, selection_mva=poisson['selection_mva'] ) ) 

    def floatParameters(self, paramlist):
        for p in paramlist:
            if p not in self.float_parameters:
                logger.info(f"Float parameter {p} not known.")
            self.float_parameters[p] = True
            logger.info(f"Float parameter {p}.")

    def get_p_mc_dcr_calibrated(self, name, selection):
        # Create a cache if it doesn't exist already
        if not hasattr(self, "_dcr_cache"):
            self._dcr_cache = {}
        key = (name, selection)
        if key not in self._dcr_cache:
            p_mc = self.h5s[name][selection]["MultiClassifier_predict"]
            p_mc_dcr = p_mc / p_mc.sum(axis=1, keepdims=True)
            self._dcr_cache[key] = self.calibrate_dcr(selection, p_mc_dcr)
        return self._dcr_cache[key]

    def dSigmaOverDSigmaSM_h5( self, name, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_tes=0, nu_jes=0, nu_met=0):
        # Multiclassifier
        #p_mc = self.h5s[name][selection]["MultiClassifier_predict"]

        p_mc_dcr_calibrated = self.get_p_mc_dcr_calibrated(name, selection)

        # Multiply the ICP to the CSI
        icp_summand = {}
        for t in ['htautau', 'ztautau', 'ttbar', 'diboson']:
            if t in self.icps and selection in self.icps[t]:
                icp_summand[t] = self.icps[t][selection].log_predict((nu_tes, nu_jes, nu_met))
            else:
                icp_summand[t] = 1
        logger.debug( "icp_summand %s tes %3.2f jes %3.2f met %3.2f "%(selection, nu_tes, nu_jes, nu_met)+" ".join( ["%s: %4.3f"%(t,icp_summand[t]) for t in self.cfg['Tasks'] if t!='MultiClassifier'] ) )

        # htautau
        DA_pnn_htautau = self.h5s[name][selection]["htautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_htautau = self.models['htautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        log_p_pnn_htautau = icp_summand['htautau'] + np.dot(DA_pnn_htautau, nu_A_htautau)

        # ztautau
        DA_pnn_ztautau = self.h5s[name][selection]["ztautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_ztautau = self.models['ztautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        log_p_pnn_ztautau = icp_summand['ztautau'] + np.dot(DA_pnn_ztautau, nu_A_ztautau)

        # ttbar
        DA_pnn_ttbar = self.h5s[name][selection]["ttbar_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_ttbar = self.models['ttbar'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        log_p_pnn_ttbar = icp_summand['ttbar'] + np.dot(DA_pnn_ttbar, nu_A_ttbar)

        # diboson
        DA_pnn_diboson = self.h5s[name][selection]["diboson_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_diboson = self.models['diboson'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        log_p_pnn_diboson= icp_summand['diboson'] + np.dot(DA_pnn_diboson, nu_A_diboson)

        # RATES
        log_f_bkg_rate = nu_bkg*np.log1p(self.alpha_bkg)
        log_f_tt_rate = nu_tt*np.log1p(self.alpha_tt)
        log_f_diboson_rate = nu_diboson*np.log1p(self.alpha_diboson)

        #p_mc_dcr = p_mc/p_mc.sum(axis=1, keepdims=True) # First divide toget DCR
        #p_mc_dcr_calibrated = self.calibrate_dcr(selection, p_mc_dcr)

        term1 = mu*(p_mc_dcr_calibrated[:, 0])*np.exp(log_p_pnn_htautau)
        term2 = (p_mc_dcr_calibrated[:, 1])*np.exp(log_f_bkg_rate + log_p_pnn_ztautau)
        term3 = (p_mc_dcr_calibrated[:, 2])*np.exp(log_f_tt_rate + log_f_bkg_rate + log_p_pnn_ttbar)
        term4 = (p_mc_dcr_calibrated[:, 3])*np.exp(log_f_diboson_rate + log_f_bkg_rate + log_p_pnn_diboson)
        return term1 + term2 + term3 + term4


        ## Find the dominant term for each event
        #max_term = np.maximum.reduce([term1, term2, term3, term4])

        ## Normalize numerator and denominator
        #numerator = (term1 + term2 + term3 + term4) / max_term
        #denominator = denominator / max_term

        #return numerator / denominator
        #### Now rewrite the return using log1p
        ####result = np.log1p(numerator / denominator - 1)

    def incS_diff_from_csis( self, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_tes=0, nu_jes=0, nu_met=0):

        # RATES
        f_bkg_rate_m1 = np.expm1(nu_bkg * np.log1p(self.alpha_bkg))
        f_tt_rate_m1 = np.expm1(nu_tt * np.log1p(self.alpha_tt))
        f_diboson_rate_m1 = np.expm1(nu_diboson * np.log1p(self.alpha_diboson))
        f_bkg_rate = f_bkg_rate_m1 + 1
        f_tt_rate = f_tt_rate_m1 + 1
        f_diboson_rate = f_diboson_rate_m1 + 1

        return \
              mu*self.csis[selection]['htautau']((nu_tes,nu_jes,nu_met)) + (mu-1)*self.csis_const[selection]['htautau'] \
            + f_bkg_rate*self.csis[selection]['ztautau']((nu_tes,nu_jes,nu_met)) + f_bkg_rate_m1*self.csis_const[selection]['ztautau'] \
            + f_tt_rate*f_bkg_rate*self.csis[selection]['ttbar']((nu_tes,nu_jes,nu_met)) + (f_tt_rate_m1+f_bkg_rate_m1+f_tt_rate_m1*f_bkg_rate_m1)*self.csis_const[selection]['ttbar'] \
            + f_diboson_rate*f_bkg_rate*self.csis[selection]['diboson']((nu_tes,nu_jes,nu_met)) + (f_diboson_rate_m1+f_bkg_rate_m1+f_diboson_rate_m1*f_bkg_rate_m1)*self.csis_const[selection]['diboson']

    def save(self, restrict_csis=[]):
        """
        Save the ML prediction to an HDF5 (.h5) file with:
          - Label (class label, if available)
          - Weight (event weights)
          - ML results (e.g. classifier predictions or parameters from PNN)
        If CSI (spline interpolation) is configured, it will also
        prepare interpolators for cross-section parameterization.

        Args:
            small (bool): If True, use a small subset of the data (for testing).
                          If False, use the complete data.
        """

        # ----------------------------------------------------------
        # Create output directory
        # ----------------------------------------------------------
        os.makedirs(self.cfg['tmp_path'], exist_ok=True)

        # ----------------------------------------------------------
        # Loop over selections and Save items
        # ----------------------------------------------------------
        for s in self.selections:
            for obj in self.cfg['Save']:
                # Figure out the HDF5 filename
                if obj == "Toy":
                    obj_fn = os.path.join(self.cfg['tmp_path'], self.cfg['Toy_name'] + '_' + s + '.h5')
                else:
                    obj_fn = os.path.join(self.cfg['tmp_path'], obj + '_' + s + '.h5')

                # Warn if the file already exists
                create_file = True
                if os.path.exists(obj_fn):
                    if self.overwrite:
                        logger.warning("Warning! Temporary file %s exists. It will be overwritten." % obj_fn)
                        os.remove(obj_fn)  # Remove the file if overwriting is allowed
                    else:
                        logger.info("Temporary file %s exists. Continue." % obj_fn)
                        create_file = False

                # Create a new HDF5 file to store the data
                if create_file:
                    if os.path.exists(obj_fn):  os.remove(obj_fn)
                    with h5py.File(obj_fn, "w",libver='latest') as h5f:
                        h5f.swmr_mode = True
                        # Save some metadata (e.g., the selection) into HDF5 attributes
                        h5f.attrs["selection"] = s

                        # Initialize HDF5 datasets with extendable dimensions
                        datasets_hephy = {
                            "Label": h5f.create_dataset("Label", (0,), maxshape=(None,), dtype=np.int32, compression="gzip", compression_opts=4),
                            "Weight": h5f.create_dataset("Weight", (0,), maxshape=(None,), dtype=np.float32, compression="gzip", compression_opts=4),
                        }

                        # For each task, check what we need to save and initialize datasets
                        for t in self.cfg['Tasks']:
                            if "save" not in self.cfg[t]:
                                continue
                            for iobj in self.cfg[t]['save']:
                                ds_name = t + '_' + iobj

                                # Use output dimension from model
                                if hasattr(self.models[t][s], "classes" ):
                                    output_dim = len(self.models[t][s].classes) #TFMC or XGBMC cases
                                else:
                                    output_dim = len(self.models[t][s].combinations) #PNN case

                                datasets_hephy[ds_name] = h5f.create_dataset(
                                    ds_name, (0, output_dim), maxshape=(None, output_dim), dtype=np.float32, compression="gzip", compression_opts=4
                                )

                            # Save the module and model path in the HDF5 attributes
                            h5f.attrs[t + "_module"] = self.cfg[t][s]["module"]
                            h5f.attrs[t + "_model_path"] = self.cfg[t][s]["model_path"]

                        # Decide how many events/batches to process
                        n_split = self.cfg['Save'][obj]['n_split'] if not self.small else 100

                        # Decide how we get the data: from training or from a toy file
                        if obj == "TrainingData":
                            data_input = self.training_data_loader(s, n_split)
                        else:
                            toy_path = os.path.join(self.cfg['Save'][obj]['dir'], s, self.cfg['Toy_name'] + '.h5')
                            data_input = self.load_toy_file(
                                filename = toy_path,
                                batch_size = self.cfg['Save'][obj]['batch_size'],
                                n_split = n_split
                            )

                        # Loop over batches of data and store the results incrementally
                        with tqdm(total=len(data_input), desc="Processing batches") as pbar:
                            for i_batch, batch in enumerate(data_input):
                                features, weights, labels = data_input.split(batch)

                                # Append labels and weights to datasets
                                datasets_hephy["Label"].resize(datasets_hephy["Label"].shape[0] + labels.shape[0], axis=0)
                                datasets_hephy["Label"][-labels.shape[0]:] = labels

                                datasets_hephy["Weight"].resize(datasets_hephy["Weight"].shape[0] + weights.shape[0], axis=0)
                                datasets_hephy["Weight"][-weights.shape[0]:] = weights

                                # For each task, produce predictions or DeltaA and write incrementally
                                for t in self.cfg['Tasks']:
                                    if "save" not in self.cfg[t]:
                                        continue

                                    for iobj in self.cfg[t]['save']:
                                        ds_name = t + '_' + iobj

                                        if iobj == "predict":
                                            pred = self.models[t][s].predict(features)
                                            #print("class_weights", self.models[t][s].class_weights, "weight_sums", self.models[t][s].weight_sums)
                                        elif iobj == "DeltaA":
                                            pred = self.models[t][s].get_DeltaA(features)
                                            print("DeltaA",t,s,self.models[t][s], np.array(pred).mean(axis=0) )
                                        else:
                                            raise Exception(
                                                f"Unsupported save type: '{iobj}'. "
                                                "Currently supported: 'predict', 'DeltaA'."
                                            )

                                        # Resize and append predictions
                                        datasets_hephy[ds_name].resize(datasets_hephy[ds_name].shape[0] + pred.shape[0], axis=0)
                                        datasets_hephy[ds_name][-pred.shape[0]:] = pred

                                # Check for early stopping conditions
                                if self.small or (
                                    self.cfg['Save'][obj]['max_n_batch'] > -1 and
                                    i_batch >= self.cfg['Save'][obj]['max_n_batch']
                                ):
                                    break

                                pbar.update(1)

                        logger.info("Saved temporary results in %s" % obj_fn)

                # ------------------------------------------------------
                # Build CSI interpolators if requested (for TrainingData)
                # ------------------------------------------------------
                if self.cfg.get("CSI") is not None and self.cfg["CSI"]["save"] and obj == "TrainingData":
                    if any( [ _s in restrict_csis for _s in self.selections ] ):
                        if s not in restrict_csis:
                            logger.info("Selection %s not among those we compute CSIs for. Continue." % s)
                            continue

                    from scipy.interpolate import RegularGridInterpolator
                    # Access the multi-class classifier predictions
                    with h5py.File(obj_fn, "r", swmr=True) as h5f:
                        gp = np.array(h5f['MultiClassifier_predict'])
                        weight = np.array(h5f["Weight"])

                        #Calibrate DCR
                        dcr_raw = gp / gp.sum(axis=1, keepdims=True) # First divide toget DCR
                        dcr_calibrated = self.calibrate_dcr(s, dcr_raw)
                        #print("dcr_raw", dcr_raw)
                        #print("dcr_calibrated", dcr_calibrated)

                        # Ensure DeltaA is a NumPy array
                        self.csis = self.csis if hasattr(self, 'csis') else {}
                        self.csis_const = self.csis_const if hasattr(self, 'csis_const') else {}
                        self.csis[s] = {}
                        self.csis_const[s] = {}

                        # Create the grid values for interpolation
                        coarse_spacing = 1
                        fine_spacing = 0.5

                        nu_tes_values = np.unique(np.concatenate([
                            np.arange(-10, 11, coarse_spacing),
                            np.arange(-3, 3 + fine_spacing, fine_spacing)]))
                        nu_jes_values = np.unique(np.concatenate([
                            np.arange(-10, 11, coarse_spacing),
                            np.arange(-3, 3 + fine_spacing, fine_spacing)]))
                        nu_met_values = np.unique(np.concatenate([
                            np.arange(0, 6, coarse_spacing),
                            np.arange(0, 3 + fine_spacing, fine_spacing)]))

                        grid_shape = (
                            len(nu_tes_values), len(nu_jes_values), len(nu_met_values))
                        base_points_flat = np.stack(
                            np.meshgrid(nu_tes_values, nu_jes_values, nu_met_values, indexing="ij"),
                            axis=-1).reshape(-1, 3)

                        # Iterate over tasks for CSI computation
                        for i_t, t in enumerate(self.cfg['Tasks'][1:]):
                            if any( [ _t in restrict_csis for _t in self.cfg['Tasks'][1:] ]):
                                if t not in restrict_csis:
                                    logger.info("Task %s not among those we compute CSIs for. Continue." %t)
                                    continue

                            # Check whether we have it already
                            pkl_filename = os.path.join(
                                self.cfg['tmp_path'], f"CSI_{s}_{t}_TrainingData.pkl")
                            if not os.path.exists( pkl_filename ) and not self.overwrite:
                                logger.info("Did not find %s. Will compute CSI." %pkl_filename)
                            if os.path.exists( pkl_filename ) and not self.overwrite:
                                logger.info("Found %s. Continue." %pkl_filename)
                                continue

                            nu_A_values = np.array([
                                self.models[t][s].nu_A(bp) for bp in base_points_flat])

                            DeltaA = np.array(h5f[t + "_DeltaA"])
                            dcr_calibrated_t = dcr_calibrated[:, i_t]

                            yield_values = np.zeros((nu_A_values.shape[0],))
                            const_value  = 0
                            batch_size = 10**5

                            # Multiply the ICP to the CSI
                            if t in self.icps and s in self.icps[t]:
                                icp_summand = np.array([self.icps[t][s].predict(tuple(bp)) for bp in base_points_flat] )
                            else:
                                icp_summand = np.ones(len(base_points_flat))

                            #print( "icp_summand", icp_summand, base_points_flat)

                            for start in tqdm(range(0, dcr_calibrated_t.shape[0], batch_size), desc=f"CSI {s} {t}"):
                                end = min(start + batch_size, dcr_calibrated_t.shape[0])

                                #batch_weighted = weight[start:end] * (gp_t[start:end] / gp_sum[start:end])
                                batch_weighted = weight[start:end] * dcr_calibrated_t[start:end]

                                exp_batch = np.expm1(np.log(icp_summand)+np.dot(DeltaA[start:end, :], nu_A_values.T))
                                yield_values += np.dot(batch_weighted, exp_batch)
                                const_value  += batch_weighted.sum()

                            data_cube = yield_values.reshape(grid_shape)
                            self.csis[s][t] = RegularGridInterpolator(
                                #(nu_tes_values, nu_jes_values, nu_met_values), data_cube, method="cubic") # Doesn't work
                                (nu_tes_values, nu_jes_values, nu_met_values), data_cube, method="quintic")

                            self.csis_const[s][t] = const_value
                            sm_index = np.where((base_points_flat == [0, 0, 0]).all(axis=1))[0][0]
                            max_ratio = (abs(yield_values/const_value )).max()
                            logger.info(f"CSI max relative shift (1 means 100%) for {s} {t}: {max_ratio:.2f}")

                            #for bp, val in zip( base_points_flat, yield_values ):
                            #    print(t, s, bp, val, self.csis[s][t](bp) )

                            if self.cfg.get("CSI", {}).get("save", False):
                                pkl_filename = os.path.join(
                                    self.cfg['tmp_path'], f"CSI_{s}_{t}_TrainingData.pkl")
                                with open(pkl_filename, 'wb') as pkl_file:
                                    pickle.dump((self.csis[s][t], self.csis_const[s][t]), pkl_file)
                                logger.info(f"CSI saved: {pkl_filename}")

        # Save Poisson observations for toy from yaml
        if "Poisson" in self.cfg:# and self.cfg["save"]:
            for name, poisson_data in self.poisson.items():
                if "ignore" in self.cfg["Poisson"][name] and self.cfg["Poisson"][name]["ignore"]: continue
                # Check whether we have it already
                pkl_filename = os.path.join(
                    self.cfg['tmp_path'], f"Poisson_{name}_Toy.pkl")
                if os.path.exists( pkl_filename ) and not self.overwrite:
                    logger.info("Found %s. Do not overwrite." %pkl_filename)
                    continue

                logger.info(f"Computing Poisson observation for {name}")
                data_input = self.training_data_loader(self.cfg["Poisson"][name]["preselection"], n_split=100)
                poisson_data['observation'] = self.Poisson_observation( data_input = data_input, selectors = poisson_data['mva_selectors'], small=self.small)
                with open(pkl_filename, 'wb') as pkl_file:
                    pickle.dump(poisson_data['observation'], pkl_file)
                    logger.info(f"Written Poisson observation to {pkl_filename}.")

    def Poisson_observation( self, data_input, selectors=[], small=False):
        result = {i_label:0 for i_label in [-1] + list(range(len(data_structure.labels))) }
        with tqdm(total=len(data_input), desc="Processing batches") as pbar:
            for i_batch, batch in enumerate(data_input):
                features, weights, labels = data_input.split(batch)
                data = np.column_stack( (features, weights, labels) )
                # Apply MVA cuts
                before = data.shape[0]
                data = reduce( lambda acc, f: f(acc), selectors, data )
                after = data.shape[0]
                #logger.debug(f"Applying MVA selectors leads to reduction from {before} to {after} counts")
                for i_label in [-1] + list(range(len(data_structure.labels))):
                    label_mask = (data[:, data_structure.label_index]==i_label)
                    result[i_label] += (data[label_mask, data_structure.weight_index]).sum()
                #weights = data[:, data_structure.weight_index]
                #result+=weights.sum()
                #print( weights.sum(), poisson_data )
                pbar.update(1)
                if small: break
        return result
            
    def penalty(self, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met):
        penalty_term = 0
        if not self.float_parameters["nu_tes"]:
            penalty_term += nu_tes**2
        if not self.float_parameters["nu_jes"]:
            penalty_term += nu_jes**2
        if not self.float_parameters["nu_met"]:
            penalty_term += nu_met**2
        if not self.float_parameters["nu_bkg"]:
            penalty_term += nu_bkg**2
        if not self.float_parameters["nu_tt"]:
            penalty_term += nu_tt**2
        if not self.float_parameters["nu_diboson"]:
            penalty_term += nu_diboson**2
        return penalty_term

    def predict(self, mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met,\
                      asimov_mu=None, asimov_nu_bkg=None, asimov_nu_tt=None, asimov_nu_diboson=None):
      import time
      # perform the calculation
      uTerm = {}

      logger.debug( f"\033[1mEvaluate at "
                f"mu={mu:6.4f}\033[0m, "
                f"nu_bkg={nu_bkg:6.4f}, "
                f"nu_tt={nu_tt:6.4f}, "
                f"nu_diboson={nu_diboson:6.4f}, "
                f"nu_tes={nu_tes:6.4f}, "
                f"nu_jes={nu_jes:6.4f}, "
                f"nu_met={nu_met:6.4f} "
            )

      # Load all toy data
      if self.toy_origin == "path":
            self.loadToyFromPath(filename=self.toy_path)
      elif self.toy_origin == "config":
          if self.cfg['Predict']['use_toy']:
              self.loadMLresults( name='Toy', filename=self.cfg['Toy_name'])
      elif self.toy_origin == "memory":
            self.loadToyFromMemory()

      # Load ML result for training data
      if not ( self.cfg.get("CSI") is not None and self.cfg["CSI"]["use"] ):
          self.loadMLresults( name='TrainingData', filename=self.cfg['Predict']['TrainingData'])

      for selection in self.selections:
        # loading CSIs
        if self.cfg.get("CSI") is not None and self.cfg["CSI"]["use"]:
            self.load_csis()

        if not ( self.cfg.get("CSI") is not None and self.cfg["CSI"]["use"] ):
            # dSoDS for training data
            weights = self.h5s['TrainingData'][selection]["Weight"]

            dSoDS_sim = self.dSigmaOverDSigmaSM_h5( 'TrainingData', selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met )
            incS_difference = (weights[:]*(1-dSoDS_sim)).sum()
        else:
            incS_difference = None

        if hasattr( self, "csis"):
            incS_difference_parametrized = -self.incS_diff_from_csis( selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met)

            # If we have the standard x-sec, we let's print the difference.
            if incS_difference is not None:
                dd = incS_difference-incS_difference_parametrized
                rel = (incS_difference-incS_difference_parametrized)/incS_difference
                logger.debug(
                    f"incS: mu={mu:6.4f}, "
                    f"nom: {incS_difference:6.4f} "
                    f"param: {incS_difference_parametrized:6.4f} "
                    f"diff: {dd:6.4f} "
                    f"rel: {rel:6.4f} "
                )
            else:
                incS_difference = incS_difference_parametrized
                logger.debug(
                    f"incS: mu={mu:6.4f}, "
                    f"param: {incS_difference_parametrized:6.4f} "
                )

        # dSoDS for toys

        if asimov_mu is None and asimov_nu_bkg is None and asimov_nu_tt is None and asimov_nu_diboson is None:
            weight_copy_method = lambda x:x
        else:
            weight_copy_method = copy.deepcopy

        if self.cfg['Predict']['use_toy']:
          dSoDS_toy = self.dSigmaOverDSigmaSM_h5( 'Toy', selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met )
          weights_toy = weight_copy_method(self.h5s['Toy'][selection]["Weight"])
        else:
          dSoDS_toy = dSoDS_sim
          weights_toy = weight_copy_method(weights)

        if asimov_mu is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['htautau']] = weights_toy[labels==data_structure.label_encoding['htautau']]*asimov_mu
            logger.debug( "Scaled labeled signal events by %4.3f" % asimov_mu )
        if asimov_nu_bkg is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            #before = weights_toy[labels!=data_structure.label_encoding['htautau']].sum()
            weights_toy[labels!=data_structure.label_encoding['htautau']] = weights_toy[labels!=data_structure.label_encoding['htautau']]*(1+self.alpha_bkg)**asimov_nu_bkg
            #after = weights_toy[labels!=data_structure.label_encoding['htautau']].sum()
            logger.debug( "Scaled labeled background events by (1+alpha_bkg)**asimov_nu_bkg with asimov_nu_bkg=%4.3f" % asimov_nu_bkg )
            #logger.debug( "Before %6.5f After %6.5f", before, after )
        if asimov_nu_tt is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['ttbar']] = weights_toy[labels==data_structure.label_encoding['ttbar']]*(1+self.alpha_tt)**asimov_nu_tt
            logger.debug( "Scaled labeled ttbar events by (1+alpha_tt)**asimov_nu_tt with asimov_nu_tt=%4.3f" % asimov_nu_tt )
        if asimov_nu_diboson is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['diboson']] = weights_toy[labels==data_structure.label_encoding['diboson']]*(1+self.alpha_diboson)**asimov_nu_diboson
            logger.debug( "Scaled labeled diboson events by (1+alpha_diboson)**asimov_nu_diboson with asimov_nu_diboson=%4.3f" % asimov_nu_diboson )

        log_term         = (weights_toy[:]*np.log1p(dSoDS_toy-1)).sum()
        uTerm[selection] = -2 *(incS_difference+log_term)
        logger.debug( f"uTerm: {selection} incS_difference: {-2*incS_difference} log_term: {-2*log_term} uTerm: {uTerm[selection]}" )

      penalty = self.penalty(nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met)

      uTerm_total = penalty + sum( uTerm.values() )

      logger.debug("Working on Poisson terms.")

      logger.debug( f"FCN: {uTerm_total:8.6f} penalty: {penalty:6.4f} " + " ".join( ["%s: %6.4f" % ( sel, uTerm[sel]) for sel in self.selections if sel in uTerm] ) )

      poisson_term = {}
      poisson_term_total = 0
      if "Poisson" in self.cfg:
        f_bkg_rate  = np.exp(nu_bkg*np.log1p(self.alpha_bkg))
        f_tt_rate   = np.exp(nu_tt*np.log1p(self.alpha_tt))
        f_diboson_rate = np.exp(nu_diboson*np.log1p(self.alpha_diboson))
        for name, poisson_data in self.poisson.items():
            if "ignore" in self.cfg["Poisson"][name] and self.cfg["Poisson"][name]["ignore"]: continue
            sigma_SM_h  = poisson_data['IC'].predict('htautau')
            sigma_SM_z  = poisson_data['IC'].predict('ztautau')
            sigma_SM_tt = poisson_data['IC'].predict('ttbar')
            sigma_SM_db = poisson_data['IC'].predict('diboson')
            sigma_SM_tot = sigma_SM_h+sigma_SM_z+sigma_SM_tt+sigma_SM_db
            
            S_h  = poisson_data['ICP']['htautau'].predict( (nu_tes, nu_jes, nu_met) )
            S_z  = poisson_data['ICP']['ztautau'].predict( (nu_tes, nu_jes, nu_met) )
            S_tt = poisson_data['ICP']['ttbar']  .predict( (nu_tes, nu_jes, nu_met) )
            S_db = poisson_data['ICP']['diboson'].predict( (nu_tes, nu_jes, nu_met) ) 

            poisson_obs = copy.deepcopy( poisson_data['observation'] )            
            if asimov_mu is not None:
                poisson_obs[data_structure.label_encoding['htautau']] *= asimov_mu
            if asimov_nu_bkg is not None:
                poisson_obs[data_structure.label_encoding['ztautau']] *= (1+self.alpha_bkg)**asimov_nu_bkg
                poisson_obs[data_structure.label_encoding['ttbar']]   *= (1+self.alpha_bkg)**asimov_nu_bkg
                poisson_obs[data_structure.label_encoding['diboson']] *= (1+self.alpha_bkg)**asimov_nu_bkg
            if asimov_nu_tt is not None:
                poisson_obs[data_structure.label_encoding['ttbar']] *= (1+self.alpha_tt)**asimov_nu_tt
            if asimov_nu_diboson is not None:
                poisson_obs[data_structure.label_encoding['dibosonbar']] *= (1+self.alpha_diboson)**asimov_nu_diboson

            poisson_observation = sum(poisson_obs.values())
            #print(name, poisson_data['observation'])
            poisson_term[name] = \
                +2*(  sigma_SM_h* ( mu*S_h - 1 )
                    + sigma_SM_z* ( f_bkg_rate*S_z - 1 )
                    + sigma_SM_tt*( f_bkg_rate*f_tt_rate*S_tt - 1 )
                    + sigma_SM_db*( f_bkg_rate*f_diboson_rate*S_db - 1 ) )\
                -2*poisson_observation*np.log( 
                    mu*S_h*sigma_SM_h/sigma_SM_tot + f_bkg_rate*(S_z*sigma_SM_z/sigma_SM_tot + f_tt_rate*S_tt*sigma_SM_tt/sigma_SM_tot + f_diboson_rate*S_db*sigma_SM_db/sigma_SM_tot)
                    )
            logger.debug( f"Poisson term {name}: -2 log (P( N={poisson_observation:.3f} | lambda={sigma_SM_tot:.3f})/P(...|SM)) = {poisson_term[name]:.3f}" ) 
   
        if self.small:
          logger.warning( "Skip Poisson term with --small." ) 
        else:
          poisson_term_total=sum( poisson_term.values() ) 

          #logger.debug( f"Total Poisson term from {len(self.cfg['Poisson'])} items: {poisson_term_total}")

      total = poisson_term_total + uTerm_total

      return total
