import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tqdm import tqdm

import logging
logger = logging.getLogger('UNC')

import numpy as np
import operator
import functools
import common.data_structure as data_structure
from data_loader.data_loader_2 import H5DataLoader
import os
import pickle
from math import ceil, sqrt
import importlib

class PhaseoutScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_lr, n_epochs, n_epochs_phaseout):
        self.initial_lr = initial_lr
        self.n_epochs = n_epochs
        self.n_epochs_phaseout = n_epochs_phaseout

    def __call__(self, epoch):
        epoch = tf.cast(epoch, tf.float32)  # Ensure epoch is a tensor of float32
        if self.n_epochs_phaseout <=0:
            # Constant learning rate case
            lr = tf.convert_to_tensor(self.initial_lr, dtype=tf.float32)
            #print(f"[PhaseoutScheduler] Epoch {epoch.numpy():.0f}: Constant LR = {lr.numpy()}")
            return lr
        elif epoch < self.n_epochs - self.n_epochs_phaseout:
            # Before phaseout starts, use the constant initial learning rate
            lr = tf.convert_to_tensor(self.initial_lr, dtype=tf.float32)
            #print(f"[PhaseoutScheduler] Epoch {epoch.numpy():.0f}: Constant LR = {lr.numpy()}")
            return lr
        else:
            # Phaseout period
            decay_start_epoch = self.n_epochs - self.n_epochs_phaseout
            decay_rate = self.initial_lr / self.n_epochs_phaseout
            lr = self.initial_lr - decay_rate * (epoch - decay_start_epoch)
            #print(f"[PhaseoutScheduler] Epoch {epoch.numpy():.0f}: Decay LR = {lr.numpy()} (decay_rate = {decay_rate})")
            return lr

class PNN:
    def __init__(self, config):
        """
        TensorFlow implementation of the PNN model.
        """
        self.config      = config
        self.config_name = config.__name__
        self.base_points   = np.array(config.base_points)
        self.n_base_points = len(self.base_points)

        self.nominal_base_point = np.array( config.nominal_base_point, dtype='float')
        self.combinations       = config.combinations
        self.parameters         = config.parameters
        self.input_dim          = config.input_dim
        # We should copy all the pieces needed at inference time to the config. Otherwise, a change to the config effects inference post-training.
        self.hidden_layers      = config.hidden_layers
        self.learning_rate      = config.learning_rate

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

        nu_mask = np.ones(len(self.base_points), bool)
        nu_mask[self.nominal_base_point_index] = 0

        # remove the nominal from the list of all the base_points
        self.masked_base_points = self.base_points[nu_mask]

        # computing base-point matrix
        C    = np.zeros( [len(self.combinations), len(self.combinations) ], dtype='float64')
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_comb1, comb1 in enumerate(self.combinations):
                for i_comb2, comb2 in enumerate(self.combinations):
                    C[i_comb1][i_comb2] += functools.reduce(operator.mul, [base_point[self.parameters.index(c)] for c in list(comb1)+list(comb2)], 1)

        assert np.linalg.matrix_rank(C)==C.shape[0], "Base point matrix does not have full rank. Check base points & combinations."

        self.CInv = np.linalg.inv(C)

        self._VKA = np.zeros( (len(self.masked_base_points), len(self.combinations)) )
        for i_base_point, base_point in enumerate(self.masked_base_points):
            for i_combination, combination in enumerate(self.combinations):
                res=1
                for var in combination:
                    res*=base_point[self.parameters.index(var)]

                self._VKA[i_base_point, i_combination ] = res

        ## Compute matrix Mkk from non-nominal base_points
        #self.MkA  = np.dot(self._VKA, self.CInv).transpose()
        #self.Mkkp = np.dot(self._VKA, self.MkA )

        self.model = self._build_model()

        # Initialize the learning rate scheduler
        lr_schedule = PhaseoutScheduler(
            initial_lr=config.learning_rate,
            n_epochs=config.n_epochs,
            n_epochs_phaseout=config.__dict__.get("n_epochs_phaseout",0),
        )

        # Initialize the optimizer with a constant learning rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        # Assign the learning rate schedule manually (if you want to control it in the loop)
        self.lr_schedule = lr_schedule

        # Create the checkpoint
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        if hasattr( config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            self.feature_means     = {i:0 for i in range(len(data_structure.feature_names))}
            self.feature_variances = {i:1 for i in range(len(data_structure.feature_names))}

    def _build_model(self):
        """Build a simple neural network for classification with batch normalization."""

        # Fetch parameters from config with defaults
        l1_reg = self.config.__dict__.get("l1_reg", 0.)  # Default value for L1: 0.01
        l2_reg = self.config.__dict__.get("l2_reg", 0.)  # Default value for L2: 0.01
        dropout_rate = self.config.__dict__.get("dropout_rate", 0.)  # Default dropout rate: 0.5
        initialize_zero = self.config.__dict__.get("initialize_zero", False)  # Whether to initialize outputs to zero

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        # Hidden layers with L1/L2 regularization and dropout
        for units in self.hidden_layers:
            model.add(
                Dense(
                    units,
                    activation=None,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                )
            )
            model.add(tf.keras.layers.Activation(self.config.__dict__.get("activation", "relu")))  # Apply activation
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))  # Apply dropout

        # Output layer
        if initialize_zero:
            # Initialize weights and biases to zero
            model.add(
                Dense(
                    self.num_outputs,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.Zeros(),  # Zero weights
                    bias_initializer=tf.keras.initializers.Zeros(),    # Zero biases
                )
            )
        else:
            # Default initialization
            model.add(
                Dense(
                    self.num_outputs,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                    activation=None,  # Default activation (no activation)
                )
            )

        return model

    def load_training_data( self, datasets_hephy, selection, process=None, n_split=10):
        self.training_data = {}
        self.process = process
        for base_point in self.base_points:
            base_point = tuple(base_point)
            values = self.config.get_alpha(base_point)
            data_loader = datasets_hephy.get_data_loader( selection=selection, values=values, process=process, selection_function=None, n_split=n_split)
            logger.info ("PNN training data: process %s Base point nu = %r, alpha = %r, file = %s"%( (process if process is not None else "combined"), base_point, values, data_loader.file_path))
            self.training_data[base_point] = data_loader

    def train_one_epoch(self, max_batch=-1, accumulate_histograms=False, rebin=1):
        """
        Train the model for one epoch using the data loader, with optional histogram accumulation.

        Parameters:
        - max_batch: int, maximum number of batches to process (default: -1, process all).
        - accumulate_histograms: bool, whether to accumulate histograms for visualization.

        Returns:
        - true_histograms, pred_histograms: dict, accumulated histograms if accumulate_histograms is True.
          Otherwise, returns None, None.
        """
        if accumulate_histograms:
            # Initialize histograms based on plot_options
            true_histograms = {}
            pred_histograms = {}
            bin_edges = {}

            for feature_name in data_structure.plot_options.keys():
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
                # make plots coarser
                n_bins = n_bins//rebin
                true_histograms[feature_name] = np.zeros((n_bins, len(self.base_points)))
                pred_histograms[feature_name] = np.zeros((n_bins, len(self.base_points)))
                bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)
        total_loss = 0.0
        i_batch = 0

        # Prepare loaders for iteration
        loaders = [
            self.training_data[tuple(base_point)]
            for base_point in self.base_points
        ]

        # Outer loop over batches
        for batches in tqdm(zip(*loaders), desc="Processing Batches"):

            with tf.GradientTape() as tape:
                # Process nominal batch
                nominal_batch = batches[self.nominal_base_point_index]
                features_nominal, weights_nominal, _ = H5DataLoader.split(nominal_batch)
                features_nominal_norm = (features_nominal - self.feature_means) / np.sqrt(self.feature_variances)
                features_nominal_tensor = tf.convert_to_tensor(features_nominal_norm, dtype=tf.float32)
                DeltaA_nominal = self.model(features_nominal_tensor, training=True)

                # Process all base points, including nominal
                for i_base_point, (base_point, batch) in enumerate(zip(self.base_points, batches)):
                    features_nu, weights_nu, _ = H5DataLoader.split(batch)
                    features_nu_norm = (features_nu - self.feature_means) / np.sqrt(self.feature_variances)
                    features_nu_tensor = tf.convert_to_tensor(features_nu_norm, dtype=tf.float32)
                    DeltaA_nu = self.model(features_nu_tensor, training=True)

                    # ICP scaling in the nu-term
                    if hasattr( self.config, "icp_predictor"):
                        bias_factor = self.config.icp_predictor(**{k:v for k,v in zip( self.parameters, self.base_points[i_base_point])}) 
                    else:
                        bias_factor = 1

                    # Compute weighted losses
                    if i_base_point != self.nominal_base_point_index:

                        loss_0 = tf.reduce_sum(
                            tf.convert_to_tensor(weights_nominal, dtype=tf.float32)
                            * tf.math.softplus(tf.linalg.matvec((DeltaA_nominal), self.VkA[i_base_point]))
                        )
                        loss_nu = 1./bias_factor*tf.reduce_sum(
                            tf.convert_to_tensor(weights_nu, dtype=tf.float32)
                            * tf.math.softplus(-tf.linalg.matvec((DeltaA_nu), self.VkA[i_base_point]))
                        )
                        loss = loss_0 + loss_nu
                        loss -= (np.sum(weights_nominal) + np.sum(weights_nu)) * tf.math.log(2.0)
                        total_loss += loss

                    # Accumulate histograms
                    if accumulate_histograms:
                        for feature_idx, feature_name in enumerate(data_structure.feature_names):
                            feature_values_nominal = features_nominal[:, feature_idx]
                            feature_values_nu = features_nu[:, feature_idx]
                            n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                            # make plots coarser
                            n_bins = n_bins//rebin

                            # Compute true probabilities for nu
                            true_histogram_nu, _ = np.histogram(
                                feature_values_nu,
                                bins=bin_edges[feature_name],
                                weights=weights_nu
                            )
                            true_histograms[feature_name][:, i_base_point] += true_histogram_nu

                            # Compute predicted probabilities for nu
                            pred_histogram_nu, _ = np.histogram(
                                feature_values_nominal,
                                bins=bin_edges[feature_name],
                                weights=weights_nominal * bias_factor * np.exp(
                                    tf.linalg.matvec(DeltaA_nominal, self.VkA[i_base_point]).numpy()
                                )
                            )
                            pred_histograms[feature_name][:, i_base_point] += pred_histogram_nu

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            i_batch += 1
            if max_batch > 0 and i_batch >= max_batch:
                break

        logger.info(f"Epoch loss: {total_loss:.4f}")

        if accumulate_histograms:
            return true_histograms, pred_histograms
        else:
            return None, None

    def nu_A(self, nu):
        return np.array( [ functools.reduce(operator.mul, [nu[self.parameters.index(c)] for c in list(comb)], 1) for comb in self.combinations] )

    def predict( self, features, nu):
        if hasattr( self.config, "icp_predictor"):
            bias_factor = self.config.icp_predictor(**{k:v for k,v in zip( self.parameters, nu)}) 
        else:
            bias_factor = 1

        #print( "bias_factor", bias_factor )

        DeltaA = self.model( tf.convert_to_tensor(
            (features - self.feature_means) / np.sqrt(self.feature_variances), dtype=tf.float32), training=False)
        return bias_factor*np.exp(np.dot(DeltaA.numpy(), self.nu_A(nu) ))

    def get_bias( self):
        if hasattr( self.config, "icp") and self.config.icp is not None:
            # Attention. No guarantee that ICP and PNN are trained with the same base-points. Have to be careful! We can have inconsistent definitions of nu in ICP and PNN!!
            bias = np.dot( self.config.icp.nu_A(base_point), self.config.icp.DeltaA ) 
        else:
            bias = 0
        return bias

    def get_DeltaA( self, features):
        DeltaA = self.model( tf.convert_to_tensor(
            (features - self.feature_means) / np.sqrt(self.feature_variances), dtype=tf.float32), training=False)
        return DeltaA

    def save(self, save_dir, epoch):
        """
        Save the model, optimizer state, and config module name to a file.

        Parameters:
        - save_dir: str, directory to save the checkpoints (e.g., 'models/test').
        - epoch: int, the current epoch number (used as the checkpoint filename).
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Write checkpoint and update the metadata file
        checkpoint_path = os.path.join(save_dir, str(epoch))
        self.checkpoint.write(checkpoint_path)

        # Manually create the 'checkpoint' metadata file
        with open(os.path.join(save_dir, 'checkpoint'), 'w') as f:
            f.write(f'model_checkpoint_path: "{checkpoint_path}"\n')

        logger.info(f"Model checkpoint and config saved for epoch {epoch} in {save_dir}.")

        _config     = self.config
        self.config = None

        _checkpoint     = self.checkpoint
        _optimizer      = self.optimizer
        _model          = self.model
        self.checkpoint = None  
        self.optimizer  = None  
        self.model      = None  

        config_path = os.path.join(save_dir, "config.pkl")

        with open(config_path, "wb") as f:
            #pickle.dump(self.config_name, f)
            pickle.dump(self, f)

        self.checkpoint = _checkpoint
        self.config     = _config
        self.optimizer  = _optimizer
        self.model      = _model

    @classmethod
    def load(cls, save_dir):
        """
        Class method to load a saved TFMC instance from the latest checkpoint.
        Handles corrupted or missing config.pkl files gracefully.
        """

        # Load the config module name from the pickle file
        config_path = os.path.join(save_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                old_instance = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Failed to load config.pkl due to corruption: {e}")

        # Dynamically import the config module
        # FIXME we have some old trainings where I only stored the config_name
        config = importlib.import_module(old_instance.config_name if not type(old_instance)==str else old_instance)

        # Create a new TFMC instance
        new_instance = cls(config=config)

        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if not latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in directory: {save_dir}")

        # Restore the model and optimizer state
        new_instance.checkpoint.restore(latest_checkpoint).expect_partial()
        # FIXME:
        if not type(old_instance)==str:
            new_instance.config_name = old_instance.config_name
            new_instance.VkA         = old_instance.VkA
            new_instance.feature_means     = old_instance.feature_means
            new_instance.feature_variances = old_instance.feature_variances

            logger.info(f"Model and config loaded from {latest_checkpoint} with config {old_instance.config_name}.")

        return new_instance

    def plot_convergence_root(self, true_histograms, pred_histograms, epoch, output_path, feature_names, rebin=1):
        """
        Plot and save the convergence visualization for all features in one canvas using ROOT.

        Parameters:
        - true_histograms: dict, true class probabilities accumulated over bins.
        - pred_histograms: dict, predicted class probabilities accumulated over bins.
        - epoch: int, current epoch number.
        - output_path: str, directory to save the ROOT files.
        - feature_names: list of str, feature names for the x-axis.
        """
        import ROOT
        ROOT.gStyle.SetOptStat(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()

        os.makedirs(output_path, exist_ok=True)

        num_features = len(feature_names)
        num_base_points = len(self.base_points)  # Use base points instead of classes

        for normalized in [False, True]:
            if normalized:
                # Normalize histograms relative to the true nominal distribution
                for feature_name in true_histograms.keys():
                    # Get the true nominal distribution
                    true_nominal = true_histograms[feature_name][:, self.nominal_base_point_index]
                    true_nominal = np.where(true_nominal == 0, 1, true_nominal)  # Avoid division by zero

                    # Normalize true histograms
                    for i_base_point in range(len(self.base_points)):
                        true_histograms[feature_name][:, i_base_point] /= true_nominal

                    # Normalize predicted histograms
                    for i_base_point in range(len(self.base_points)):
                        pred_histograms[feature_name][:, i_base_point] /= true_nominal

            # Calculate grid size, adding one pad for the legend
            total_pads = num_features + 1
            grid_size_x = int(ceil(sqrt(total_pads)))
            grid_size_y = int(ceil(total_pads / grid_size_x))
            canvas = ROOT.TCanvas("c_convergence", "Convergence Plot", 500 * grid_size_x, 500 * grid_size_y)
            canvas.Divide(grid_size_x, grid_size_y)

            colors = data_structure.colors[:num_base_points] 
            colors[self.nominal_base_point_index] = ROOT.kBlack

            stuff = []  # Prevent ROOT objects from being garbage collected

            # Loop through each feature
            for feature_idx, feature_name in enumerate(feature_names):
                pad = canvas.cd(feature_idx + 1)
                pad.SetTicks(1, 1)
                pad.SetBottomMargin(0.15)
                pad.SetLeftMargin(0.15)

                pad.SetLogy(not normalized and data_structure.plot_options[feature_name]['logY'])

                # Determine the maximum y-value for scaling
                max_y = 0
                for i_base_point in range(num_base_points):
                    max_y = max(
                        max_y,
                        true_histograms[feature_name][:, i_base_point].max(),
                        pred_histograms[feature_name][:, i_base_point].max(),
                    )

                # Fetch binning and axis title from plot_options
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]["binning"]

                # make plots coarser
                n_bins = n_bins//rebin
                
                x_axis_title = data_structure.plot_options[feature_name]["tex"]

                if normalized: 
                    min_y = max(0, 1-(1.2 * max_y-1))
                else:
                    min_y = 0

                max_y = 1.2*max_y

                # Use y_ratio_range if provided
                if normalized:
                    min_y, max_y = data_structure.plot_options[feature_name].get('y_ratio_range', [min_y, max_y])
                
                h_frame = ROOT.TH2F(
                    f"h_frame_{feature_name}",
                    f";{x_axis_title};Probability",
                    n_bins, x_min, x_max,
                    100, min_y, max_y,
                )
                h_frame.GetYaxis().SetTitleOffset(1.3)
                h_frame.Draw()
                stuff.append(h_frame)

                # Loop through base points to create and style histograms
                for i_base_point, base_point in enumerate(self.base_points):
                    # True probabilities (dashed)
                    h_true = ROOT.TH1F(
                        f"h_true_{feature_name}_{i_base_point}",
                        f"{feature_name} (true {base_point})",
                        n_bins, x_min, x_max,
                    )
                    for i, y in enumerate(true_histograms[feature_name][:, i_base_point]):
                        h_true.SetBinContent(i + 1, y)

                    h_true.SetLineColor(colors[i_base_point % len(colors)])
                    h_true.SetLineStyle(2)  # Dashed
                    h_true.SetLineWidth(2)
                    h_true.Draw("HIST SAME")
                    stuff.append(h_true)

                    # Predicted probabilities (solid)
                    h_pred = ROOT.TH1F(
                        f"h_pred_{feature_name}_{i_base_point}",
                        f"{feature_name} (pred {base_point})",
                        n_bins, x_min, x_max,
                    )
                    for i, y in enumerate(pred_histograms[feature_name][:, i_base_point]):
                        h_pred.SetBinContent(i + 1, y)

                    h_pred.SetLineColor(colors[i_base_point % len(colors)])
                    h_pred.SetLineStyle(1)  # Solid
                    h_pred.SetLineWidth(2)
                    h_pred.Draw("HIST SAME")
                    stuff.append(h_pred)

            # Legend in the last pad
            legend_pad_index = num_features + 1
            canvas.cd(legend_pad_index)

            legend = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
            legend.SetNColumns( 1+num_base_points//20 )
            legend.SetBorderSize(0)
            legend.SetShadowColor(0)

            # Create dummy histograms for legend
            dummy_true = []
            dummy_pred = []

            for i_base_point, base_point in enumerate(self.base_points):
                # Dummy histogram for true probabilities
                hist_true = ROOT.TH1F(f"dummy_true_{i_base_point}", "", 1, 0, 1)
                hist_true.SetLineColor(colors[i_base_point % len(colors)])
                hist_true.SetLineStyle(2)  # Dashed
                hist_true.SetLineWidth(2)
                dummy_true.append(hist_true)

                # Dummy histogram for predicted probabilities
                hist_pred = ROOT.TH1F(f"dummy_pred_{i_base_point}", "", 1, 0, 1)
                hist_pred.SetLineColor(colors[i_base_point % len(colors)])
                hist_pred.SetLineStyle(1)  # Solid
                hist_pred.SetLineWidth(2)
                dummy_pred.append(hist_pred)

                # Add entries to the legend
                legend.AddEntry(hist_true, f"{base_point} (true)", "l")
                legend.AddEntry(hist_pred, f"{base_point} (pred)", "l")

            legend.Draw()
            stuff.extend(dummy_true + dummy_pred)

            tex = ROOT.TLatex()
            tex.SetNDC()
            tex.SetTextSize(0.07)
            tex.SetTextAlign(11)  # Align right

            lines = [(0.3, 0.95, f"Epoch = {epoch:04d}")]
            drawObjects = [tex.DrawLatex(*line) for line in lines]
            for o in drawObjects:
                o.Draw()

            # Save the canvas
            norm = "norm_" if normalized else ""
            output_file = os.path.join(output_path, f"{norm}epoch_{epoch:04d}.png")
            for fmt in ["png"]:
                canvas.SaveAs(output_file.replace(".png", f".{fmt}"))

            logger.info(f"Saved convergence plot for epoch {epoch} to {output_file}.")

