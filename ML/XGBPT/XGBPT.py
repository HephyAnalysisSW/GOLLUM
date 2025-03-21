import xgboost as xgb
import numpy as np
import os
import pickle
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure
import common.user as user
from tqdm import tqdm
from math import ceil, sqrt

import logging
logger = logging.getLogger('UNC')

import operator
import functools
from data_loader.data_loader_2 import H5DataLoader
import importlib

class Objective:
    def __init__(self):
        # Key: DMatrix handle (int), Value: (measureTag_array, measureIndex_array)
        self.metadata = {}

    def set_metadata(self, dmatrix, measure_tag, measure_idx):
        """Store the row-wise metadata (arr of shape (n_rows,)) for the given DMatrix."""
        handle_id = dmatrix.handle.value  # an integer that uniquely identifies the DMatrix
        self.metadata[handle_id] = (measure_tag, measure_idx)

    def __call__(self, preds, dtrain):
        """
        Custom objective function.
        
        Args:
            preds: shape (n_rows,) – the raw predictions (scores) from the model
            dtrain: the DMatrix object

        Returns:
            (grad, hess): each array of shape (n_rows,)
        """
        handle_id = dtrain.handle.value
        # retrieve measureTag, measureIndex for these rows
        measure_tag, measure_idx = self.metadata[handle_id]

        # For illustration, let's do something trivial:
        #   - If measureTag=0 => L0 =  log(1 + exp(preds)) 
        #   - If measureTag=1 => L1 =  log(1 + exp(-preds))
        # Then measureIndex is just an integer ID we might use if we want more logic.

        # We'll build grad, hess row by row. 
        # But let's do vectorized. For measureTag=0:
        # grad0 = e^preds / (1 + e^preds)
        # hess0 = e^preds / (1 + e^preds)^2
        # For measureTag=1:
        # grad1 = - 1/(1 + e^preds)
        # hess1 = e^preds / (1 + e^preds)^2
        
        preds = np.array(preds, copy=False)  # shape (n_rows,)
        exp_preds = np.exp(preds)

        print( "preds", preds.shape, preds)
        print( "measure_tag", measure_tag.shape, measure_tag)
        print( "measure_idx", measure_idx.shape, measure_idx)

        grad = np.zeros_like(preds, dtype=np.float32)
        hess = np.zeros_like(preds, dtype=np.float32)

        mask0 = (measure_tag == 0)
        mask1 = (measure_tag == 1)

        # measureTag=0 => log(1 + e^preds)
        logistic0 = exp_preds[mask0] / (1.0 + exp_preds[mask0])
        grad[mask0] = logistic0
        hess[mask0] = logistic0 * (1.0 - logistic0)

        # measureTag=1 => log(1 + e^-preds)
        logistic1 = 1.0 / (1.0 + exp_preds[mask1])  # = e^-pred / (1 + e^-pred)
        grad[mask1] = -logistic1
        hess[mask1] = logistic1 * (1.0 - logistic1)

        return grad, hess

objective = Objective()

class XGBPT:
    def __init__(self, config = None, model_dir=None):
        """
        TensorFlow implementation of the XGBPT model.
        """
    
        if config is not None:
            self.config      = config
            self.config_name = config.__name__
            self.base_points   = np.array(config.base_points)
            self.n_base_points = len(self.base_points)
            self.model_dir   = config.model_dir if hasattr( config, "model_dir" ) else model_dir
            if self.model_dir is None:
                raise RuntimeError( "Please provide a model_dir either in the config or as an argument" )

            self.nominal_base_point = np.array( config.nominal_base_point, dtype='float')
            self.combinations       = config.combinations
            self.parameters         = config.parameters

            self.num_outputs  = len(self.combinations)

            # Base point matrix
            self.VkA  = np.zeros( [len(self.base_points), len(self.combinations) ], dtype=np.float32)
            for i_base_point, base_point in enumerate(self.base_points):
                for i_comb1, comb1 in enumerate(self.combinations):
                    self.VkA[i_base_point][i_comb1] += functools.reduce(operator.mul, [base_point[self.parameters.index(c)] for c in list(comb1)], 1)

            # Dissect inputs into nominal sample and variied
            nominal_base_point_index = np.where(np.all(self.base_points==self.nominal_base_point,axis=1))[0]
            assert len(nominal_base_point_index)>0, "Could not find nominal base %r point in training data keys %r"%( self.nominal_base_point, self.base_points)
            self.nominal_base_point_index = nominal_base_point_index[0]
            self.nominal_base_point_key   = tuple(self.nominal_base_point)

        elif model_dir is not None:
            self.model_dir = model_dir

        else:
            raise Exception("Please provide a config.")

        #nu_mask = np.ones(len(self.base_points), bool)
        #nu_mask[self.nominal_base_point_index] = 0

        ## remove the nominal from the list of all the base_points
        #self.masked_base_points = self.base_points[nu_mask]

        ## computing base-point matrix
        #C    = np.zeros( [len(self.combinations), len(self.combinations) ], dtype='float64')
        #for i_base_point, base_point in enumerate(self.masked_base_points):
        #    for i_comb1, comb1 in enumerate(self.combinations):
        #        for i_comb2, comb2 in enumerate(self.combinations):
        #            C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[self.parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

        #assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

        #self.CInv = np.linalg.inv(C)

        #self._VKA = np.zeros( (len(self.masked_base_points), len(self.combinations)) )
        #for i_base_point, base_point in enumerate(self.masked_base_points):
        #    for i_combination, combination in enumerate(self.combinations):
        #        res=1
        #        for var in combination:
        #            res*=base_point[self.parameters.index(var)]

        #        self._VKA[i_base_point, i_combination ] = res

    def load_training_data( self, datasets, selection, process=None, n_split=10):
        self.training_data = {}
        self.process = process
        for base_point in self.base_points:
            base_point = tuple(base_point)
            values = self.config.get_alpha(base_point)
            data_loader = datasets.get_data_loader( selection=selection, values=values, process=process, selection_function=None, n_split=n_split)
            print ("XGBPT training data: process %s Base point nu = %r, alpha = %r, file = %s"%( (process if process is not None else "combined"), base_point, values, data_loader.file_path))
            self.training_data[base_point] = data_loader

    def train(self, max_batch=-1, every=-1, plot_directory=None):
        """
        Training loop that:
         - Assumes each base_point has exactly the same number of batches
         - Zips the data loaders so we get one batch from each base point simultaneously
         - Nominal base point is identified by self.nominal_base_point_index
         - Replicates nominal rows for each non-nominal base point in that batch
         - Stores measureTag=0 or 1, and measureIndex (integer) instead of the nu-value
         - Does exactly 1 XGBoost iteration per combined batch

        We rely on "same number of batches" across all loaders, so we don't do while-loops or resets.
        """

        import xgboost as xgb
        import numpy as np
        from tqdm import tqdm
        
        if not hasattr(self, "training_data"):
            raise ValueError("No training data found. Call `load_training_data` first.")

        # XGBoost hyperparams
        self.params = {
            'eta':               self.config.learning_rate,
            'max_depth':         self.config.max_depth,
            'subsample':         self.config.subsample,
            'colsample_bytree':  self.config.colsample_bytree,
            'lambda':            self.config.l2_reg,   # L2
            'alpha':             self.config.l1_reg,   # L1
            'seed':              self.config.seed,
            'tree_method':       getattr(self.config, 'tree_method', 'auto'),
        }
        self.num_boost_round = getattr(self.config, "num_boost_round", 100)

        # Identify all base points (including nominal).
        # e.g. base_points = array([[-1.], [ 0.], [ 1.]])
        bp_array = self.base_points  # shape (N_base_points, ?)
        # nominal_idx = e.g. 1 if (0,) is the second row in bp_array
        nominal_idx = self.nominal_base_point_index

        # Build a list of data loaders in the same order as bp_array
        # We'll do: data_loaders[i] corresponds to base_points[i]
        data_loaders = []
        for i, bp in enumerate(bp_array):
            bp_key = tuple(bp)
            dl = self.training_data[bp_key]
            data_loaders.append(dl)

        # We'll now zip them => each iteration yields a tuple of N_base_points batches
        # that correspond to the same "batch index" across each base point.
        # We assume that each loader has the same # of batches.
        loader_zipped = zip(*data_loaders)  # no while loop needed

        # Outer "epoch" loop
        for epoch in range(self.start_epoch, self.num_boost_round):

            # For each batch in the data loaders
            for i_batch, batch_tuple in enumerate(tqdm(loader_zipped, desc=f"Epoch {epoch+1}/{self.num_boost_round} Batches")):
                if max_batch > 0 and i_batch >= max_batch:
                    break

                # batch_tuple is something like: (batch_for_bp0, batch_for_bp1, batch_for_bp2, ...)
                # If we have e.g. 3 base points, it's length 3

                # 1) Identify nominal batch
                nominal_batch = batch_tuple[nominal_idx]
                data_nominal, w_nominal, _ = data_loaders[nominal_idx].split(nominal_batch)
                # data_nominal: shape (N_nominal, n_features)
                # w_nominal: shape (N_nominal,)

                # 2) Gather non-nominal batches
                X_measure0   = []
                W_measure0   = []
                measureTag0  = []
                measureIndex0= []
                
                X_measureNu   = []
                W_measureNu   = []
                measureTagNu  = []
                measureIndexNu= []

                # We'll replicate the nominal data for each non-nominal index j != nominal_idx
                # and also gather the actual measure-nu data from j != nominal_idx
                for j, b_j in enumerate(batch_tuple):
                    if j == nominal_idx:
                        # skip because we already splitted nominal above
                        continue

                    # parse the batch from base_point j
                    data_j, w_j, _ = data_loaders[j].split(b_j)

                    # measureIndex j => integer that identifies base_points[j]
                    # measureTag=1 => measure-nu row
                    # measureTag=0 => measure-0 row (the nominal replicate)

                    # replicate nominal if both sets are non-empty
                    if data_nominal.shape[0] > 0 and data_j.shape[0] > 0:
                        X_measure0.append(data_nominal)
                        W_measure0.append(w_nominal)
                        measureTag0.append(np.zeros_like(w_nominal))      # measureTag=0
                        measureIndex0.append(np.full_like(w_nominal, j))  # measureIndex=j => the "other" measure

                    # measure-nu rows
                    if data_j.shape[0] > 0:
                        X_measureNu.append(data_j)
                        W_measureNu.append(w_j)
                        measureTagNu.append(np.ones_like(w_j))            # measureTag=1
                        measureIndexNu.append(np.full_like(w_j, j))       # measureIndex=j

                # 3) Concatenate measure-0 arrays
                if len(X_measure0) > 0:
                    X_measure0   = np.concatenate(X_measure0, axis=0)
                    W_measure0   = np.concatenate(W_measure0, axis=0)
                    measureTag0  = np.concatenate(measureTag0, axis=0)
                    measureIndex0= np.concatenate(measureIndex0, axis=0)
                else:
                    # empty
                    X_measure0   = np.empty((0, data_nominal.shape[1]))
                    W_measure0   = np.empty((0,))
                    measureTag0  = np.empty((0,))
                    measureIndex0= np.empty((0,))

                # 4) Concatenate measure-nu arrays
                if len(X_measureNu) > 0:
                    X_measureNu   = np.concatenate(X_measureNu, axis=0)
                    W_measureNu   = np.concatenate(W_measureNu, axis=0)
                    measureTagNu  = np.concatenate(measureTagNu, axis=0)
                    measureIndexNu= np.concatenate(measureIndexNu, axis=0)
                else:
                    X_measureNu   = np.empty((0, data_nominal.shape[1]))
                    W_measureNu   = np.empty((0,))
                    measureTagNu  = np.empty((0,))
                    measureIndexNu= np.empty((0,))

                # 5) Merge measure-0 & measure-nu => X_big
                X_big = np.concatenate([X_measure0, X_measureNu], axis=0)
                W_big = np.concatenate([W_measure0, W_measureNu], axis=0)
                measureTag_arr   = np.concatenate([measureTag0,   measureTagNu], axis=0)
                measureIndex_arr = np.concatenate([measureIndex0, measureIndexNu], axis=0)

                if X_big.shape[0] == 0:
                    # Possibly no data for this batch => skip
                    continue

                # 7) Build DMatrix
                dummy_labels = np.zeros(X_big.shape[0], dtype=np.float32)
                dtrain = xgb.DMatrix(X_big, label=dummy_labels, weight=W_big)

                # store measureTag_arr & measureIndex_arr
                #dtrain.set_float_info("measureTag",   measureTag_arr.astype(np.float32))
                #dtrain.set_float_info("measureIndex", measureIndex_arr.astype(np.float32))

                objective.set_metadata(dtrain, measureTag_arr, measureIndex_arr)

                # 8) One boosting iteration
                if not hasattr(self, "model") or self.model is None:
                    self.model = xgb.train(
                        params=self.params,
                        dtrain=dtrain,
                        num_boost_round=1,
                        obj=objective,
                        xgb_model=None
                    )
                else:
                    self.model = xgb.train(
                        params=self.params,
                        dtrain=dtrain,
                        num_boost_round=1,
                        obj=objective,
                        xgb_model=self.model
                    )

            # end of batch loop
            # Optionally do plotting or logging every `every` epochs
            if every>0 and (epoch+1) % every == 0:
                pass

            # Save checkpoint
            self.save(epoch=epoch+1)

    def nu_A(self, nu):
        return np.array( [ functools.reduce(operator.mul, [nu[self.parameters.index(c)] for c in list(comb)], 1) for comb in self.combinations] )

#    def predict( self, features, nu):
#        if hasattr( self.config, "icp_predictor"):
#            bias_factor = self.config.icp_predictor(**{k:v for k,v in zip( self.parameters, nu)}) 
#        else:
#            bias_factor = 1
#
#        DeltaA = self.model( tf.convert_to_tensor(
#            (features - self.feature_means) / np.sqrt(self.feature_variances), dtype=tf.float32), training=False)
#        return bias_factor*np.exp(np.dot(DeltaA.numpy(), self.nu_A(nu) ))
#
#    def get_bias( self):
#        if hasattr( self.config, "icp") and self.config.icp is not None:
#            # Attention. No guarantee that ICP and PNN are trained with the same base-points. Have to be careful! We can have inconsistent definitions of nu in ICP and PNN!!
#            bias = np.dot( self.config.icp.nu_A(base_point), self.config.icp.DeltaA ) 
#        else:
#            bias = 0
#        return bias
#
#    def get_DeltaA( self, features):
#        DeltaA = self.model( tf.convert_to_tensor(
#            (features - self.feature_means) / np.sqrt(self.feature_variances), dtype=tf.float32), training=False)
#        return DeltaA
#
#
    def save(self, epoch):
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.json")
        metadata_path = os.path.join(self.model_dir, f"model_metadata_{epoch:04d}.pkl")

        os.makedirs(self.model_dir, exist_ok=True)

        if self.model:
            self.model.save_model(model_path)
            metadata = {
                'epoch': epoch,
                'model_dir': self.model_dir,
                'num_boost_round': self.num_boost_round,
                'params':self.params,
                'config_name' : self.config_name,
                'base_points' : self.base_points,
                'n_base_points' : self.n_base_points,
                'nominal_base_point' : self.nominal_base_point, 
                'combinations' : self.combinations,
                'parameters' : self.parameters,
                'num_outputs' : self.num_outputs,
                'VkA' : self.VkA, 
                'nominal_base_point_index' : self.nominal_base_point_index,
                'nominal_base_point_key' : self.nominal_base_point_key,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Model and metadata saved for epoch {epoch}")
        else:
            raise Exception("Model is not trained yet!")

    @classmethod
    def load(cls, model_dir, return_epoch=False):
        model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".json")]
        if not model_files:
            return None, 0

        model_files.sort()
        last_model_file = model_files[-1]
        epoch = int(last_model_file.split('_')[1].split('.')[0])
        model_path = os.path.join(model_dir, last_model_file)
        metadata_path = os.path.join( model_dir, last_model_file.replace('model', 'model_metadata').replace('.json', '.pkl'))
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        instance = cls(config=None, model_dir = model_dir)
        instance.model = xgb.Booster()
        instance.model.load_model(model_path)

        # Here we set the meta_data
        instance.params = metadata['params']
        instance.config_name = metadata['config_name']
        instance.base_points = metadata['base_points']
        instance.n_base_points = metadata['n_base_points']
        instance.nominal_base_point = metadata['nominal_base_point']
        instance.combinations = metadata['combinations']
        instance.parameters = metadata['parameters']
        instance.num_outputs = metadata['num_outputs']
        instance.VkA = metadata['VkA']
        instance.nominal_base_point_index = metadata['nominal_base_point_index']
        instance.nominal_base_point_key = metadata['nominal_base_point_key']

        instance.config = importlib.import_module(instance.config_name)

        logger.info(f"Model and metadata loaded from {model_path}, epoch {epoch}")
        if return_epoch:
            return instance, epoch
        else:
            return instance

#
#    def plot_convergence_root(self, true_histograms, pred_histograms, epoch, output_path, feature_names, rebin=1):
#        """
#        Plot and save the convergence visualization for all features in one canvas using ROOT.
#
#        Parameters:
#        - true_histograms: dict, true class probabilities accumulated over bins.
#        - pred_histograms: dict, predicted class probabilities accumulated over bins.
#        - epoch: int, current epoch number.
#        - output_path: str, directory to save the ROOT files.
#        - feature_names: list of str, feature names for the x-axis.
#        """
#        import ROOT
#        ROOT.gStyle.SetOptStat(0)
#        dir_path = os.path.dirname(os.path.realpath(__file__))
#        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
#        ROOT.setTDRStyle()
#
#        os.makedirs(output_path, exist_ok=True)
#
#        num_features = len(feature_names)
#        num_base_points = len(self.base_points)  # Use base points instead of classes
#
#        for normalized in [False, True]:
#            if normalized:
#                # Normalize histograms relative to the true nominal distribution
#                for feature_name in true_histograms.keys():
#                    # Get the true nominal distribution
#                    true_nominal = true_histograms[feature_name][:, self.nominal_base_point_index]
#                    true_nominal = np.where(true_nominal == 0, 1, true_nominal)  # Avoid division by zero
#
#                    # Normalize true histograms
#                    for i_base_point in range(len(self.base_points)):
#                        true_histograms[feature_name][:, i_base_point] /= true_nominal
#
#                    # Normalize predicted histograms
#                    for i_base_point in range(len(self.base_points)):
#                        pred_histograms[feature_name][:, i_base_point] /= true_nominal
#
#            # Calculate grid size, adding one pad for the legend
#            total_pads = num_features + 1
#            grid_size_x = int(ceil(sqrt(total_pads)))
#            grid_size_y = int(ceil(total_pads / grid_size_x))
#            canvas = ROOT.TCanvas("c_convergence", "Convergence Plot", 500 * grid_size_x, 500 * grid_size_y)
#            canvas.Divide(grid_size_x, grid_size_y)
#
#            colors = data_structure.colors[:num_base_points] 
#            colors[self.nominal_base_point_index] = ROOT.kBlack
#
#            stuff = []  # Prevent ROOT objects from being garbage collected
#
#            # Loop through each feature
#            for feature_idx, feature_name in enumerate(feature_names):
#                pad = canvas.cd(feature_idx + 1)
#                pad.SetTicks(1, 1)
#                pad.SetBottomMargin(0.15)
#                pad.SetLeftMargin(0.15)
#
#                pad.SetLogy(not normalized and data_structure.plot_options[feature_name]['logY'])
#
#                # Determine the maximum y-value for scaling
#                max_y = 0
#                for i_base_point in range(num_base_points):
#                    max_y = max(
#                        max_y,
#                        true_histograms[feature_name][:, i_base_point].max(),
#                        pred_histograms[feature_name][:, i_base_point].max(),
#                    )
#
#                # Fetch binning and axis title from plot_options
#                n_bins, x_min, x_max = data_structure.plot_options[feature_name]["binning"]
#
#                # make plots coarser
#                n_bins = n_bins//rebin
#                
#                x_axis_title = data_structure.plot_options[feature_name]["tex"]
#
#                if normalized: 
#                    min_y = max(0, 1-(1.2 * max_y-1))
#                else:
#                    min_y = 0
#
#                max_y = 1.2*max_y
#
#                # Use y_ratio_range if provided
#                if normalized:
#                    min_y, max_y = data_structure.plot_options[feature_name].get('y_ratio_range', [min_y, max_y])
#                
#                h_frame = ROOT.TH2F(
#                    f"h_frame_{feature_name}",
#                    f";{x_axis_title};Probability",
#                    n_bins, x_min, x_max,
#                    100, min_y, max_y,
#                )
#                h_frame.GetYaxis().SetTitleOffset(1.3)
#                h_frame.Draw()
#                stuff.append(h_frame)
#
#                # Loop through base points to create and style histograms
#                for i_base_point, base_point in enumerate(self.base_points):
#                    # True probabilities (dashed)
#                    h_true = ROOT.TH1F(
#                        f"h_true_{feature_name}_{i_base_point}",
#                        f"{feature_name} (true {base_point})",
#                        n_bins, x_min, x_max,
#                    )
#                    for i, y in enumerate(true_histograms[feature_name][:, i_base_point]):
#                        h_true.SetBinContent(i + 1, y)
#
#                    h_true.SetLineColor(colors[i_base_point % len(colors)])
#                    h_true.SetLineStyle(2)  # Dashed
#                    h_true.SetLineWidth(2)
#                    h_true.Draw("HIST SAME")
#                    stuff.append(h_true)
#
#                    # Predicted probabilities (solid)
#                    h_pred = ROOT.TH1F(
#                        f"h_pred_{feature_name}_{i_base_point}",
#                        f"{feature_name} (pred {base_point})",
#                        n_bins, x_min, x_max,
#                    )
#                    for i, y in enumerate(pred_histograms[feature_name][:, i_base_point]):
#                        h_pred.SetBinContent(i + 1, y)
#
#                    h_pred.SetLineColor(colors[i_base_point % len(colors)])
#                    h_pred.SetLineStyle(1)  # Solid
#                    h_pred.SetLineWidth(2)
#                    h_pred.Draw("HIST SAME")
#                    stuff.append(h_pred)
#
#            # Legend in the last pad
#            legend_pad_index = num_features + 1
#            canvas.cd(legend_pad_index)
#
#            legend = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
#            legend.SetNColumns( 1+num_base_points//20 )
#            legend.SetBorderSize(0)
#            legend.SetShadowColor(0)
#
#            # Create dummy histograms for legend
#            dummy_true = []
#            dummy_pred = []
#
#            for i_base_point, base_point in enumerate(self.base_points):
#                # Dummy histogram for true probabilities
#                hist_true = ROOT.TH1F(f"dummy_true_{i_base_point}", "", 1, 0, 1)
#                hist_true.SetLineColor(colors[i_base_point % len(colors)])
#                hist_true.SetLineStyle(2)  # Dashed
#                hist_true.SetLineWidth(2)
#                dummy_true.append(hist_true)
#
#                # Dummy histogram for predicted probabilities
#                hist_pred = ROOT.TH1F(f"dummy_pred_{i_base_point}", "", 1, 0, 1)
#                hist_pred.SetLineColor(colors[i_base_point % len(colors)])
#                hist_pred.SetLineStyle(1)  # Solid
#                hist_pred.SetLineWidth(2)
#                dummy_pred.append(hist_pred)
#
#                # Add entries to the legend
#                legend.AddEntry(hist_true, f"{base_point} (true)", "l")
#                legend.AddEntry(hist_pred, f"{base_point} (pred)", "l")
#
#            legend.Draw()
#            stuff.extend(dummy_true + dummy_pred)
#
#            tex = ROOT.TLatex()
#            tex.SetNDC()
#            tex.SetTextSize(0.07)
#            tex.SetTextAlign(11)  # Align right
#
#            lines = [(0.3, 0.95, f"Epoch = {epoch:04d}")]
#            drawObjects = [tex.DrawLatex(*line) for line in lines]
#            for o in drawObjects:
#                o.Draw()
#
#            # Save the canvas
#            norm = "norm_" if normalized else ""
#            output_file = os.path.join(output_path, f"{norm}epoch_{epoch:04d}.png")
#            for fmt in ["png"]:
#                canvas.SaveAs(output_file.replace(".png", f".{fmt}"))
#
#            print(f"Saved convergence plot for epoch {epoch} to {output_file}.")
#
