import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '.')
import importlib
import h5py
import numpy as np
from tqdm import tqdm
import networks.Models as ms
from data_loader.data_loader_2 import H5DataLoader
import common.user as user
import common.data_structure as data_structure
import pickle
import copy

import logging
logger = logging.getLogger('UNC')

class Inference:
    def __init__(self, cfg, small=False, overwrite=False):
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

        # Initialize attributes
        self.training_data = {}
        self.models        = {}
        self.teststat      = {}
        self.selections    = self.cfg['Selections']
        self.small         = small
        self.overwrite     = overwrite
        self.h5s           = {}

        # Load models and parameters
        self.load_models()
        self.alpha_bkg     = self.cfg['Parameters']['alpha_bkg']
        self.alpha_tt      = self.cfg['Parameters']['alpha_tt']
        self.alpha_diboson = self.cfg['Parameters']['alpha_diboson']

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


    def training_data_loader(self, selection, n_split):
        """
        Load training data for a given selection.
        
        Args:
            selection (str): Selection criteria.
            n_split (int): Number of splits for the dataset.

        Returns:
            DataLoader: The loaded training data.
        """
        import common.datasets as datasets
        d = datasets.get_data_loader(selection=selection, n_split=n_split)
        logger.info("Training data loaded for selection: {}".format(selection))
        return d

    def load_toy_file(self, filename, batch_size, n_split):
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
            selection_function = None,
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
        for t in self.cfg['Tasks']:
            assert h5f.attrs[t + "_module"] == self.cfg[t][selection]['module'], \
                "Task {} selection {}: inconsistent module! H5: {} -- Config: {}".format(
                    t, selection, h5f.attrs[t + "_module"], self.cfg[t][selection]['module'])
            assert h5f.attrs[t + "_model_path"] == self.cfg[t][selection]['model_path'], \
                "Task {} selection {}: inconsistent model path! H5 {} -- Config {}".format(
                    t, selection, h5f.attrs[t + "_model_path"], self.cfg[t][selection]['model_path'])
        return h5f

    def loadMLresults(self, name, filename, selection, ignore_done=False):
        """
        Load machine learning results from an HDF5 file.
        
        Args:
            name (str): Name of the results.
            filename (str): Base name of the HDF5 file.
            selection (str): Selection criteria.
            ignore_done (bool): Whether to ignore already loaded results.
        """
        h5_filename = os.path.join( self.cfg['tmp_path'], filename + '_' + selection + '.h5' )
        assert os.path.exists(h5_filename), "File {} does not exist! Try running the save mode first.".format(h5_filename)

        if not ignore_done and name in self.h5s and selection in self.h5s[name]:
            return  # Results already loaded, skip

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
 
    def dSigmaOverDSigmaSM_h5( self, name, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_tes=0, nu_jes=0, nu_met=0):
        # Multiclassifier
        p_mc = self.h5s[name][selection]["MultiClassifier_predict"]
  
        # htautau
        DA_pnn_htautau = self.h5s[name][selection]["htautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_htautau = self.models['htautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        p_pnn_htautau = np.exp( np.dot(DA_pnn_htautau, nu_A_htautau))
  
        # ztautau
        DA_pnn_ztautau = self.h5s[name][selection]["ztautau_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_ztautau = self.models['ztautau'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        p_pnn_ztautau = np.exp( np.dot(DA_pnn_ztautau, nu_A_ztautau))
  
        # ttbar
        DA_pnn_ttbar = self.h5s[name][selection]["ttbar_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_ttbar = self.models['ttbar'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        p_pnn_ttbar = np.exp( np.dot(DA_pnn_ttbar, nu_A_ttbar))
  
        # diboson
        DA_pnn_diboson = self.h5s[name][selection]["diboson_DeltaA"] # <- this should be Nx9, 9 numbers per event
        nu_A_diboson = self.models['diboson'][selection].nu_A((nu_tes,nu_jes,nu_met)) # <- use this
        p_pnn_diboson= np.exp( np.dot(DA_pnn_diboson, nu_A_diboson))
  
        # RATES
        f_bkg_rate = (1+self.alpha_bkg)**nu_bkg
        f_tt_rate = (1+self.alpha_tt)**nu_tt
        f_diboson_rate = (1+self.alpha_diboson)**nu_diboson
  
        return ((mu*p_mc[:,0]*p_pnn_htautau + p_mc[:,1]*f_bkg_rate*p_pnn_ztautau + p_mc[:,2]*f_tt_rate*f_bkg_rate*p_pnn_ttbar + p_mc[:,3]*f_diboson_rate*f_bkg_rate*p_pnn_diboson) / p_mc[:,:].sum(axis=1))

        ## Compute all terms in numerator
        #term1 = mu * p_mc[:, 0] * p_pnn_htautau
        #term2 = p_mc[:, 1] * f_bkg_rate * p_pnn_ztautau
        #term3 = p_mc[:, 2] * f_tt_rate * f_bkg_rate * p_pnn_ttbar
        #term4 = p_mc[:, 3] * f_diboson_rate * f_bkg_rate * p_pnn_diboson

        ## Find the dominant term for each event
        #denominator = p_mc.sum(axis=1)
        #max_term = np.maximum.reduce([term1, term2, term3, term4])

        ## Normalize numerator and denominator
        #numerator = (term1 + term2 + term3 + term4) / max_term
        #denominator = denominator / max_term

        #return numerator / denominator
        ## Now rewrite the return using log1p
        ##result = np.log1p(numerator / denominator - 1)
 
    def incS_diff_from_csis( self, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_tes=0, nu_jes=0, nu_met=0):
  
        # RATES
        f_bkg_rate = (1+self.alpha_bkg)**nu_bkg
        f_tt_rate = (1+self.alpha_tt)**nu_tt
        f_diboson_rate = (1+self.alpha_diboson)**nu_diboson
  
        return \
              mu*self.csis[selection]['htautau']((nu_tes,nu_jes,nu_met)) + (mu-1)*self.csis_const[selection]['htautau'] \
            + f_bkg_rate*self.csis[selection]['ztautau']((nu_tes,nu_jes,nu_met)) + (f_bkg_rate-1)*self.csis_const[selection]['ztautau'] \
            + f_tt_rate*f_bkg_rate*self.csis[selection]['ttbar']((nu_tes,nu_jes,nu_met)) + (f_tt_rate*f_bkg_rate-1)*self.csis_const[selection]['ttbar'] \
            + f_diboson_rate*f_bkg_rate*self.csis[selection]['diboson']((nu_tes,nu_jes,nu_met)) + (f_diboson_rate*f_bkg_rate-1)*self.csis_const[selection]['diboson']

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
                        datasets = {
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

                                datasets[ds_name] = h5f.create_dataset(
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
                                toy_path,
                                self.cfg['Save'][obj]['batch_size'],
                                n_split
                            )

                        # Loop over batches of data and store the results incrementally
                        with tqdm(total=len(data_input), desc="Processing batches") as pbar:
                            for i_batch, batch in enumerate(data_input):
                                features, weights, labels = data_input.split(batch)

                                # Append labels and weights to datasets
                                datasets["Label"].resize(datasets["Label"].shape[0] + labels.shape[0], axis=0)
                                datasets["Label"][-labels.shape[0]:] = labels

                                datasets["Weight"].resize(datasets["Weight"].shape[0] + weights.shape[0], axis=0)
                                datasets["Weight"][-weights.shape[0]:] = weights

                                # For each task, produce predictions or DeltaA and write incrementally
                                for t in self.cfg['Tasks']:
                                    if "save" not in self.cfg[t]:
                                        continue

                                    for iobj in self.cfg[t]['save']:
                                        ds_name = t + '_' + iobj

                                        if iobj == "predict":
                                            pred = self.models[t][s].predict(features)
                                        elif iobj == "DeltaA":
                                            pred = self.models[t][s].get_DeltaA(features)
                                        else:
                                            raise Exception(
                                                f"Unsupported save type: '{iobj}'. "
                                                "Currently supported: 'predict', 'DeltaA'."
                                            )

                                        # Resize and append predictions
                                        datasets[ds_name].resize(datasets[ds_name].shape[0] + pred.shape[0], axis=0)
                                        datasets[ds_name][-pred.shape[0]:] = pred

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
                                if os.path.exists( pkl_filename ) and not self.overwrite:
                                    logger.info("Found %s. Continue." %pkl_filename)
                                    continue
                         
                            nu_A_values = np.array([
                                self.models[t][s].nu_A(bp) for bp in base_points_flat])

                            DeltaA = np.array(h5f[t + "_DeltaA"])
                            gp_t, gp_sum = gp[:, i_t], gp.sum(axis=1)

                            yield_values = np.zeros((nu_A_values.shape[0],))
                            const_value  = 0 
                            batch_size = 10**5

                            for start in tqdm(range(0, gp_t.shape[0], batch_size), desc=f"CSI {s} {t}"):
                                end = min(start + batch_size, gp_t.shape[0])
                                batch_weighted = weight[start:end] * (gp_t[start:end] / gp_sum[start:end])
                                exp_batch = np.expm1(np.dot(DeltaA[start:end, :], nu_A_values.T))
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

                            if self.cfg.get("CSI", {}).get("save", False):
                                pkl_filename = os.path.join(
                                    self.cfg['tmp_path'], f"CSI_{s}_{t}_TrainingData.pkl")
                                with open(pkl_filename, 'wb') as pkl_file:
                                    pickle.dump((self.csis[s][t], self.csis_const[s][t]), pkl_file)
                                logger.info(f"CSI saved: {pkl_filename}")

    def penalty(self, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met):
          return nu_bkg**2+nu_tt**2+nu_diboson**2+nu_tes**2+nu_jes**2+nu_met**2
  
    def predict(self, mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met,\
                      asimov_mu=None, asimov_nu_bkg=None, asimov_nu_tt=None, asimov_nu_diboson=None):
      import time
      # perform the calculation
      uTerm = {}

      logger.debug( f"Evaluate at"
                f"nu_bkg={nu_bkg:6.4f}, "
                f"nu_tt={nu_tt:6.4f}, "
                f"nu_diboson={nu_diboson:6.4f}, "
                f"nu_tes={nu_tes:6.4f}, "
                f"nu_jes={nu_jes:6.4f}, "
                f"nu_met={nu_met:6.4f}, "
            )
      for selection in self.selections:
  
        # Load ML result for training data
        self.loadMLresults( name='TrainingData', filename=self.cfg['Predict']['TrainingData'], selection=selection)

        # loading CSIs
        if self.cfg.get("CSI") is not None and self.cfg["CSI"]["use"]: 
            self.load_csis()

        # Load ML result for toy
        if self.cfg['Predict']['use_toy']:
            self.loadMLresults( name='Toy', filename=self.cfg['Toy_name'], selection=selection)
  
        # dSoDS for training data
        weights = self.h5s['TrainingData'][selection]["Weight"]

        if not ( self.cfg.get("CSI") is not None and self.cfg["CSI"]["use"] ):
            dSoDS_sim = self.dSigmaOverDSigmaSM_h5( 'TrainingData',selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met )
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
        if self.cfg['Predict']['use_toy']:
          dSoDS_toy = self.dSigmaOverDSigmaSM_h5( 'Toy',selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met )
          weights_toy = copy.deepcopy(self.h5s['Toy'][selection]["Weight"])
        else:
          dSoDS_toy = dSoDS_sim
          weights_toy = copy.deepcopy(weights)

        if asimov_mu is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['htautau']] = weights_toy[labels==data_structure.label_encoding['htautau']]*asimov_mu
            logger.debug( "Scaled labeled signal events by %4.3f" % asimov_mu )
        if asimov_nu_bkg is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels!=data_structure.label_encoding['htautau']] = weights_toy[labels!=data_structure.label_encoding['htautau']]*(1+self.alpha_bkg)**asimov_nu_bkg
            logger.debug( "Scaled labeled background events by (1+alpha_bkg)**asimov_nu_bkg with asimov_nu_bkg=%4.3f" % asimov_nu_bkg )
        if asimov_nu_tt is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['ttbar']] = weights_toy[labels==data_structure.label_encoding['ttbar']]*(1+self.alpha_ttbar)**asimov_nu_tt
            logger.debug( "Scaled labeled ttbar events by (1+alpha_ttbar)**asimov_nu_tt with asimov_nu_tt=%4.3f" % asimov_nu_tt )
        if asimov_nu_diboson is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['diboson']] = weights_toy[labels==data_structure.label_encoding['diboson']]*(1+self.alpha_diboson)**asimov_nu_diboson
            logger.debug( "Scaled labeled diboson events by (1+alpha_diboson)**asimov_nu_diboson with asimov_nu_diboson=%4.3f" % asimov_nu_diboson )
 
        #uTerm[selection] = -2 *(incS_difference+(weights_toy[:]*np.log1p(dSoDS_toy-1)).sum())
        log_term         = (weights_toy[:]*np.log(dSoDS_toy)).sum()
        uTerm[selection] = -2 *(incS_difference+log_term)
        logger.debug( f"uTerm: {selection} incS_difference: {-2*incS_difference} log_term: {-2*log_term} uTerm: {uTerm[selection]}" ) 

      penalty = self.penalty(nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met)

      uTerm_total = penalty + sum( uTerm.values() )      

      logger.debug( f"FCN: {uTerm_total:8.6f} penalty: {penalty:6.4f} " + " ".join( ["%s: %6.4f" % ( sel, uTerm[sel]) for sel in self.selections ] ) ) 

      return uTerm_total 
  
    def clossMLresults(self):
        logger.warning( "Warning. clossMLresults does nothing" )
        return
        #for n in list(self.h5s):
        #  for s in list(self.h5s[n]):
        #    # self.h5s[n][s].close()
        #    del self.h5s[n][s]
        #  del self.h5s[n]
