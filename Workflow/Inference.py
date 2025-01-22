import sys
import os
sys.path.insert( 0, '..')
sys.path.insert( 0, '.')
import importlib
import yaml
import h5py
import numpy as np
from tqdm import tqdm
import networks.Models as ms
from data_loader.data_loader_2 import H5DataLoader
import common.user as user
import common.data_structure as data_structure
import pickle
import copy

class Inference:
    def __init__(self, cfg_path, small=False, overwrite=False):
        """
        Initialize the Inference object.
        Load configuration, models, and parameters.
        
        Args:
            cfg_path (str): Path to the configuration file.
            small (bool): Whether to use a smaller dataset.
        """
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        print("Config loaded from {}".format(cfg_path))

        # Ensure required sections are defined in the config
        assert "Tasks" in self.cfg, "Section Tasks not defined in config!"
        assert "Selections" in self.cfg, "Section Selections not defined in config!"

        # Initialize attributes
        self.training_data = {}
        self.toys          = {}
        self.models        = {}
        self.teststat      = {}
        self.selections    = self.cfg['Selections']
        self.small         = small
        self.overwrite     = overwrite
        self.h5s           = {}

        self.output_directory = os.path.join(user.output_directory, 
            os.path.basename( cfg_path ).replace(".yaml", ""), 
            'tmp_data' + ('_small' if self.small else ''))

        # Load models and parameters
        self.load_models()
        self.alpha_bkg     = self.cfg['Parameters']['alpha_bkg']
        self.alpha_tt      = self.cfg['Parameters']['alpha_tt']
        self.alpha_diboson = self.cfg['Parameters']['alpha_diboson']

        # ----------------------------------------------------------
        # Determine whether we do CSIs 
        # ----------------------------------------------------------
        self.do_csis = self.cfg.get('CSI')
        if self.do_csis is None:
            self.do_csis = False
        if self.do_csis:
            try:
                self.load_csis()
                print("Loaded existing CSIs.")
            except (IOError, AssertionError):
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
        print("Training data loaded for selection: {}".format(selection))
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
        print("Toy loaded from {}.".format(filename))
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
            print(f"File {filename} blocked.")
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
        h5_filename = os.path.join( self.output_directory,
            filename + '_' + selection + '.h5'
        )
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
        print("ML results {} with {} loaded from {}".format(name, selection, h5_filename))

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
                print("Task {} selection {}: Module {} loaded with model path {}.".format(
                    t, s, self.cfg[t][s]["module"], self.cfg[t][s]["model_path"]))

    def load_csis(self):
        """
        Load the csis (interpolators).
        
        """
        # Already loaded
        if hasattr( self, "csis" ): return

        pkl_filename = os.path.join( self.output_directory, "CSI_TrainingData.pkl" )
        assert os.path.exists(pkl_filename), "CSIs file {} does not exist!".format(pkl_filename)

        with open(pkl_filename, 'rb') as f:
            self.csis = pickle.load(f)

        print(f"CSIs loaded from {pkl_filename}.")
 
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
  
        #return (mu*p_mc[:,0]/(p_mc[:,1:].sum(axis=1)) + 1)*p_pnn_jes
        return ((mu*p_mc[:,0]*p_pnn_htautau + p_mc[:,1]*f_bkg_rate*p_pnn_ztautau + p_mc[:,2]*f_tt_rate*f_bkg_rate*p_pnn_ttbar + p_mc[:,3]*f_diboson_rate*f_bkg_rate*p_pnn_diboson) / p_mc[:,:].sum(axis=1))

    def incS_from_csis( self, selection, mu=1, nu_bkg=0, nu_tt=0, nu_diboson=0, nu_tes=0, nu_jes=0, nu_met=0):
  
        # RATES
        f_bkg_rate = (1+self.alpha_bkg)**nu_bkg
        f_tt_rate = (1+self.alpha_tt)**nu_tt
        f_diboson_rate = (1+self.alpha_diboson)**nu_diboson
  
        return mu*self.csis[selection]['htautau']((nu_tes,nu_jes,nu_met)) + f_bkg_rate*self.csis[selection]['ztautau']((nu_tes,nu_jes,nu_met)) + f_tt_rate*f_bkg_rate*self.csis[selection]['ttbar']((nu_tes,nu_jes,nu_met)) + f_diboson_rate*f_bkg_rate*self.csis[selection]['diboson']((nu_tes,nu_jes,nu_met))

    def save(self):
        """
        Save the ML prediction to an HDF5 (.h5) file with:
          - Label (class label, if available)
          - Weight (event weights)
          - ML results (e.g. classifier predictions or parameters from PNN)
        If CSI (cubic spline interpolation) is configured, it will also
        prepare interpolators for cross-section parameterization.

        Args:
            small (bool): If True, use a small subset of the data (for testing).
                          If False, use the complete data.
        """

        # ----------------------------------------------------------
        # Create output directory
        # ----------------------------------------------------------
        os.makedirs(self.output_directory, exist_ok=True)

        # ----------------------------------------------------------
        # Loop over selections and Save items
        # ----------------------------------------------------------
        for s in self.selections:
            for obj in self.cfg['Save']:

                # Figure out the HDF5 filename
                if obj == "Toy":
                    obj_fn = os.path.join(self.output_directory, self.cfg['Toy_name'] + '_' + s + '.h5')
                else:
                    obj_fn = os.path.join(self.output_directory, obj + '_' + s + '.h5')

                # Warn if the file already exists
                if os.path.exists(obj_fn):
                    if self.overwrite:
                        print("Warning! Temporary file %s exists. It will be overwritten." % obj_fn)
                    else:
                        print("Temporary file %s exists. Continue." % obj_fn)
                        continue

                # ------------------------------------------------------
                # Create a new HDF5 file to store the data
                # ------------------------------------------------------
                with h5py.File(obj_fn, "w") as h5f:
                    # Prepare dictionaries for storing data in memory before writing
                    datasets = {
                        "Label": [],
                        "Weight": [],
                    }

                    # Save some metadata (e.g. the selection) into HDF5 attributes
                    h5f.attrs["selection"] = s

                    # For each task, check what we need to save and store module/model path info
                    for t in self.cfg['Tasks']:
                        if "save" not in self.cfg[t]:
                            continue
                        # Initialize dataset(s) for whatever is under 'save'
                        for iobj in self.cfg[t]['save']:
                            datasets[t + '_' + iobj] = []

                        # Save the module and model path in the HDF5 attributes
                        h5f.attrs[t + "_module"] = self.cfg[t][s]["module"]
                        h5f.attrs[t + "_model_path"] = self.cfg[t][s]["model_path"]

                    # Decide how many events/batches to process
                    n_split = self.cfg['Save'][obj]['n_split'] if not self.small else 100

                    # ------------------------------------------------------
                    # Decide how we get the data: from training or from a toy file
                    # ------------------------------------------------------
                    if obj == "TrainingData":
                        data_input = self.training_data_loader(s, n_split)
                    else:
                        toy_path = os.path.join(self.cfg['Save'][obj]['dir'], s, self.cfg['Toy_name'] + '.h5')
                        data_input = self.load_toy_file(
                            toy_path,
                            self.cfg['Save'][obj]['batch_size'],
                            n_split
                        )

                    # ------------------------------------------------------
                    # Loop over batches of data and store the results
                    # ------------------------------------------------------
                    for i_batch, batch in enumerate(data_input):
                        features, weights, labels = data_input.split(batch)

                        ## For toy data, set labels to -1 (since real labels may not exist)
                        #if obj != "TrainingData":
                        #    nevts = features.shape[0]
                        #    labels = np.array([-1] * nevts)
                        ## Robert: We need the labels to modify event weights for Asimov limits with modified nu_bkg etc.

                        # Store labels and weights
                        datasets["Label"].append(labels)
                        datasets["Weight"].append(weights)

                        # For each task, produce the predictions or DeltaA
                        for t in self.cfg['Tasks']:
                            if "save" not in self.cfg[t]:
                                continue

                            for iobj in self.cfg[t]["save"]:
                                if iobj == "predict":
                                    pred = self.models[t][s].predict(features)
                                    datasets[t + '_' + iobj].append(pred)
                                elif iobj == "DeltaA":
                                    DA = self.models[t][s].get_DeltaA(features)
                                    datasets[t + '_' + iobj].append(DA)
                                else:
                                    raise Exception(
                                        "Unsupported save type: '%s'. "
                                        "Currently supported: 'predict', 'DeltaA'." % iobj
                                    )

                        # If 'small' is True or we have reached a user-specified batch limit, break early
                        if self.small or (
                            self.cfg['Save'][obj]['max_n_batch'] > -1 and 
                            i_batch >= self.cfg['Save'][obj]['max_n_batch']
                        ):
                            break

                    # ------------------------------------------------------
                    # Concatenate all batches and write them to the HDF5 file
                    # ------------------------------------------------------
                    for ds_name, ds_content in datasets.items():
                        # Ensure all datasets are concatenated to a single NumPy array
                        ds_merged = np.concatenate(ds_content, axis=0)
                        h5f.create_dataset(
                            ds_name,
                            data=ds_merged,
                            compression="gzip",
                            compression_opts=4  # 1=fastest, 9=smallest
                        )
                        datasets[ds_name]=ds_merged

                    print("Saved temporary results in {}".format(obj_fn))

                    # ------------------------------------------------------
                    # Build CSI interpolators if requested (for TrainingData)
                    # ------------------------------------------------------
                    if self.do_csis and obj == "TrainingData":
                        from scipy.interpolate import RegularGridInterpolator

                        # Access the multi-class classifier predictions
                        gp     = datasets['MultiClassifier_predict']
                        weight = datasets["Weight"]

                        # We start computing csis
                        if not hasattr( self, "csis" ):
                            self.csis = {}

                        self.csis[s] = {}

                        # Ensure NumPy arrays
                        if isinstance(gp, list):
                            gp = np.concatenate(gp, axis=0)
                        if isinstance(weight, list):
                            weight = np.concatenate(weight, axis=0)

                        # Hardcode the 3D grid definition with both coarse and finer spacing
                        coarse_spacing = 1
                        fine_spacing = 0.5

                        # Coarse grid
                        nu_tes_values_coarse = np.arange(-10, 11, coarse_spacing)
                        nu_jes_values_coarse = np.arange(-10, 11, coarse_spacing)
                        nu_met_values_coarse = np.arange(0, 6, coarse_spacing)

                        # Fine grid (only within specified ranges)
                        nu_tes_values_fine = np.arange(-3, 3 + fine_spacing, fine_spacing)
                        nu_jes_values_fine = np.arange(-3, 3 + fine_spacing, fine_spacing)
                        nu_met_values_fine = np.arange(0, 3 + fine_spacing, fine_spacing)

                        # Combine coarse and fine grids, ensuring no duplicates
                        nu_tes_values = np.unique(np.concatenate([nu_tes_values_coarse, nu_tes_values_fine]))
                        nu_jes_values = np.unique(np.concatenate([nu_jes_values_coarse, nu_jes_values_fine]))
                        nu_met_values = np.unique(np.concatenate([nu_met_values_coarse, nu_met_values_fine]))

                        # Create the combined grid
                        grid_shape = (len(nu_tes_values), len(nu_jes_values), len(nu_met_values))
                        nu_tes_grid, nu_jes_grid, nu_met_grid = np.meshgrid(nu_tes_values, nu_jes_values, nu_met_values, indexing="ij")

                        # Flatten the grid for easier matrix operations
                        base_points_flat = np.stack([nu_tes_grid.ravel(), nu_jes_grid.ravel(), nu_met_grid.ravel()], axis=-1)

                        # Iterate over tasks and datasets
                        for i_t, t in enumerate(self.cfg['Tasks'][1:]):
                            # Precompute nu_A(base_points) for all base points
                            nu_A_values = np.array([self.models[t][s].nu_A(base_point) for base_point in base_points_flat]) 

                            # Ensure DeltaA is a NumPy array
                            DeltaA = np.array(datasets[t + "_DeltaA"])  # Convert to NumPy array if it's a list

                            gp_t   = gp[:, i_t]  # Shape: (172734,)
                            gp_sum = gp[:,:].sum(axis=1)

                            # Dynamically determine batch size
                            batch_size = 10**5 #max(1, int((DeltaA.shape[0] * nu_A_values.shape[0]) / 10**5))
                            #print(f"Using batch size: {batch_size}")

                            # Initialize yield_values to accumulate results
                            yield_values = np.zeros((nu_A_values.shape[0],), dtype=float)  # Shape: (726,)

                            # Process events in batches
                            for start in tqdm(range(0, gp_t.shape[0], batch_size), desc="Processing batches", unit="batch"):
                                end = min(start + batch_size, gp_t.shape[0])

                                # Slice the current batch
                                gp_batch_weighted = weight[start:end]*(gp_t[start:end]/gp_sum[start:end])  # Shape: (batch_size,)
                                #gp_batch_weighted = gp_t[start:end]  # Shape: (batch_size,)
                                DeltaA_batch = DeltaA[start:end, :]  # Shape: (batch_size, <other_dimensions>)
                                #print("DeltaA_batch shape:", DeltaA_batch.shape)
                                #print("nu_A_values.T shape:", nu_A_values.T.shape)
                                #print("gp_batch_weightd.shape:", gp_batch_weighted.shape)
                                # Compute the dot product and exponentiation for the batch
                                exp_values_batch = np.exp(np.dot(DeltaA_batch, nu_A_values.T))  # Shape: (batch_size, 726)

                                # Weighted summation for the batch
                                yield_values += np.dot(gp_batch_weighted, exp_values_batch)  # Shape: (726,)

                            # Reshape yield values back into the grid
                            data_cube = yield_values.reshape(grid_shape)  # Shape: (11, 11, 6)

                            # Create the interpolator
                            self.csis[s][t] = RegularGridInterpolator(
                                (nu_tes_values, nu_jes_values, nu_met_values), data_cube, method="quintic",
                            )

                            # As a check, compute the max ratio w.r.t. (0,0,0)
                            sm_index = list(map(tuple, base_points_flat)).index((0, 0, 0))
                            max_ratio = max(
                                abs(yield_values[i_bp] / yield_values[sm_index]) for i_bp in range(len(base_points_flat))
                            )
                            print("CSI: Maximum ratio within training boundaries for %s %s: %3.2f" % (s, t, max_ratio))
                            # Save the CSI interpolator(s) to a pickle

                            #for base_point in base_points_flat:
                            #    index=list(map(tuple, base_points_flat)).index(tuple(base_point))
                            #    print( s,t,base_point, yield_values[index], self.csis[s][t](base_point) )


        if self.do_csis:
            pkl_filename = os.path.join ( self.output_directory, "CSI_TrainingData.pkl" )
            pickle.dump(self.csis, open(pkl_filename, 'wb'))
            print("CSI: Written %s" % pkl_filename)

    def penalty(self, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met):
          return nu_bkg**2+nu_tt**2+nu_diboson**2+nu_tes**2+nu_jes**2+nu_met**2
  
    def predict(self, mu, nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met,\
                      asimov_mu=None, asimov_nu_bkg=None, asimov_nu_ttbar=None, asimov_nu_diboson=None):
      import time
      # perform the calculation
      uTerm = 0
      for selection in self.selections:
  
        # Load ML result for training data
        self.loadMLresults( name='TrainingData', filename=self.cfg['Predict']['TrainingData'], selection=selection)

        # loading CSIs
        if self.do_csis: 
            self.load_csis()

        # Load ML result for toy
        if self.cfg['Predict']['use_toy']:
            self.loadMLresults( name='Toy', filename=self.cfg['Toy_name'], selection=selection)
  
        # dSoDS for training data
        weights = self.h5s['TrainingData'][selection]["Weight"]
        dSoDS_sim = self.dSigmaOverDSigmaSM_h5( 'TrainingData',selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met )
        incS_difference = (weights[:]*(1-dSoDS_sim)).sum()

        if hasattr( self, "csis"):
            incS_difference_parametrized = -self.incS_from_csis( selection, mu=mu, nu_bkg=nu_bkg, nu_tt=nu_tt, nu_diboson=nu_diboson, nu_tes=nu_tes, nu_jes=nu_jes, nu_met=nu_met ) + self.incS_from_csis(selection) 
            dd = incS_difference-incS_difference_parametrized
            rel = (incS_difference-incS_difference_parametrized)/incS_difference
            sm_value = weights[:].sum()
            sm_value_param = self.incS_from_csis(selection)
            print(
                f"incS: mu {mu:6.4f} "
                f"nu_bkg {nu_bkg:6.4f} "
                f"nu_tt {nu_tt:6.4f} "
                f"nu_diboson {nu_diboson:6.4f} "
                f"nu_tes {nu_tes:6.4f} "
                f"nu_jes {nu_jes:6.4f} "
                f"nu_met {nu_met:6.4f} "
                f"nom: {incS_difference:6.4f} "
                f"param: {incS_difference_parametrized:6.4f} "
                f"diff: {dd:6.4f} "
                f"rel: {rel:6.4f} "
                #f"SM {sm_value:6.2f} "
                #f"SM param {sm_value_param:6.2f} "
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
            print( "Scaled labeled signal events by %4.3f" % asimov_mu )
        if asimov_nu_bkg is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels!=data_structure.label_encoding['htautau']] = weights_toy[labels!=data_structure.label_encoding['htautau']]*asimov_nu_bkg
            print( "Scaled labeled background events by %4.3f" % asimov_nu_bkg )
        if asimov_nu_ttbar is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['ttbar']] = weights_toy[labels==data_structure.label_encoding['ttbar']]*asimov_nu_ttbar
            print( "Scaled labeled ttbar events by %4.3f" % asimov_ttbar )
        if asimov_nu_diboson is not None:
            labels = self.h5s['Toy'][selection]["Label"]
            weights_toy[labels==data_structure.label_encoding['diboson']] = weights_toy[labels==data_structure.label_encoding['diboson']]*asimov_nu_diboson
            print( "Scaled labeled diboson events by %4.3f" % asimov_diboson )
  
        uTerm += -2 *(incS_difference+(weights_toy[:]*np.log(dSoDS_toy)).sum())

      uTerm += self.penalty(nu_bkg, nu_tt, nu_diboson, nu_tes, nu_jes, nu_met)

      return uTerm
  
    def clossMLresults(self):
        print( "Warning. clossMLresults does nothing" )
        return
        #for n in list(self.h5s):
        #  for s in list(self.h5s[n]):
        #    # self.h5s[n][s].close()
        #    del self.h5s[n][s]
        #  del self.h5s[n]
