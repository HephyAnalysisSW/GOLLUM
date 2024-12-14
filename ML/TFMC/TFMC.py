import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from math import ceil, sqrt
import pickle
import importlib
import ROOT
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal, Constant

class PhaseoutScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_lr, n_epochs, n_epoch_phaseout):
        self.initial_lr = initial_lr
        self.n_epochs = n_epochs
        self.n_epoch_phaseout = n_epoch_phaseout

    def __call__(self, epoch):
            """
            Calculate the learning rate based on the current epoch.
            
            Parameters:
            - epoch: int or tf.Tensor, current epoch number.
            
            Returns:
            - float, learning rate for the current epoch.
            """
            epoch = tf.cast(epoch, tf.float32)  # Ensure the epoch is a float
            if epoch < self.n_epochs - self.n_epoch_phaseout:
                return tf.convert_to_tensor(self.initial_lr, dtype=tf.float32)
            else:
                decay_start_epoch = self.n_epochs - self.n_epoch_phaseout
                decay_rate = self.initial_lr / self.n_epoch_phaseout
                return self.initial_lr - decay_rate * (epoch - decay_start_epoch)
class TFMC:
    def __init__(self, config=None, input_dim=None, classes=None, hidden_layers=None, reweighting=True):
        """
        Initialize the multiclass classifier model.
        
        Parameters:
        - input_dim: int, number of features in the input data.
        - num_classes: int, number of output classes.
        """
        
        # Whether to perform class reweighting
        self.reweighting = reweighting

        if config is not None:
            self.config        = config
            self.config_name   = config.__name__
            self.input_dim     = config.input_dim 
            self.classes       = config.classes
            self.hidden_layers = config.hidden_layers
        elif (input_dim is not None) and (classes is not None) and (hidden_layers is not None):
            self.config        = None
            self.config_name   = None
            self.input_dim     = input_dim 
            self.classes       = classes
            self.hidden_layers = hidden_layers
        else:
            raise Exception("Please provide either a config or all other parameters (input_dim, classes, hidden_layers).")

        self.num_classes = len(self.classes) 

        self.model = self._build_model()

        # Integrate learning rate scheduler
        lr_schedule = PhaseoutScheduler(
            initial_lr=config.learning_rate,
            n_epochs=config.n_epochs,
            n_epoch_phaseout=config.__dict__.get("n_epoch_phaseout",0),
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.metrics = tf.keras.metrics.CategoricalAccuracy()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        if hasattr( config, "weight_sums"):
            self.weight_sums = config.weight_sums
        else:
            self.weight_sums = {i:1./self.num_classes for i in range(self.num_classes)}

        if hasattr( config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            self.feature_means     = np.array([0 for i in range(len(data_structure.feature_names))])
            self.feature_variances = np.array([1 for i in range(len(data_structure.feature_names))])

        # Scale cross sections to the same integral
        self.class_weights = [ self.weight_sums[data_structure.label_encoding[label]] for label in self.config.classes ]
        total = sum(self.class_weights)
        self.class_weights = np.array([total/self.class_weights[i] for i in range(len(self.class_weights))])
        
        print("Will scale with these factors: "+" ".join( ["%s: %3.2f"%( self.classes[i], self.class_weights[i]) for i in range( self.num_classes)]) )

    def _build_model(self):
        """Build a simple neural network for classification with L1/L2 regularization and dropout."""
        import tensorflow as tf
        from tensorflow.keras import regularizers

        # Fetch parameters from config with defaults
        l1_reg = self.config.__dict__.get("l1_reg", 0.)  # Default value for L1: 0.01
        l2_reg = self.config.__dict__.get("l2_reg", 0.)  # Default value for L2: 0.01
        dropout_rate = self.config.__dict__.get("dropout_rate", 0.)  # Default dropout rate: 0.5

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        # Hidden layers with L1/L2 regularization and dropout
        for units in self.hidden_layers:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
                )
            )
            model.add(tf.keras.layers.Activation(self.config.activation))  # Apply activation
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))  # Apply dropout

        # Output layer
        model.add(
            tf.keras.layers.Dense(
                self.num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None,
            )
        )

        return model

    def predict(self, data, ic_scaling=True):   
        res =  self.model((data - self.feature_means) / np.sqrt(self.feature_variances), training=False).numpy()
        # put back the inclusive xsec
        if ic_scaling:
            return res/self.class_weights # DCR
        else:
            return res             # LR

    def load_training_data( self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader( selection=selection, selection_function=None, n_split=n_split)

    def train_one_epoch(self, max_batch=-1, accumulate_histograms=False):
        """
        Train the model for one epoch using the data loader, with optional histogram accumulation.

        Parameters:
        - max_batch: int, maximum number of batches to process (default: -1, process all).
        - accumulate_histograms: bool, whether to accumulate histograms of true and predicted class probabilities.

        Returns:
        - true_histograms, pred_histograms: dict, accumulated histograms if accumulate_histograms is True.
          Otherwise, returns None, None.
        """
        if accumulate_histograms:
            # For histogram accumulation
            num_features = self.model.input_shape[1]
            true_histograms = {}
            pred_histograms = {}
            bin_edges = {}

            # Initialize histograms based on plot_options
            for feature_name in data_structure.plot_options.keys():
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
                true_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                pred_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)

        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        total_loss = 0.0
        total_samples = 0
        i_batch = 0

        selected_indices = [data_structure.label_encoding[label] for label in self.config.classes]

        for batch in self.data_loader:
            print(f"Batch {i_batch}")
            data, weights, raw_labels_ = self.data_loader.split(batch)

            # Filter events based on the selected classes
            mask = np.isin(raw_labels_, selected_indices) #FIXME raw_labels_ only for debugging
            data = data[mask]
            weights = weights[mask]
            raw_labels = raw_labels_[mask]

            # Normalize data
            data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

            # preprocess features, if needed
            if hasattr( self.config, "preprocessor"):
                data_norm = self.config.preprocessor( data, data_norm)

            # Apply reweighting if enabled
            if self.reweighting:
                weights = weights * self.class_weights[raw_labels.astype('int')]

            # Remap raw labels to sequential indices for the selected classes
            remapped_labels = np.array([selected_indices.index(label) for label in raw_labels])
            
            # Convert remapped labels to one-hot encoded format
            labels_one_hot = tf.keras.utils.to_categorical(remapped_labels, num_classes=self.num_classes)

            # Compute gradients and loss
            with tf.GradientTape() as tape:
                predictions = self.model(data_norm, training=True)
                loss = self.loss_fn(labels_one_hot, predictions)
                weighted_loss = tf.reduce_mean(loss * weights)

            #return data, data_norm, weights, raw_labels, raw_labels_, remapped_labels, labels_one_hot, predictions, loss

            gradients = tape.gradient(weighted_loss, self.model.trainable_variables)
            accumulated_gradients = [
                acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
            ]

            # Accumulate histograms if requested
            if accumulate_histograms:
                for feature_idx, feature_name in enumerate(data_structure.feature_names):
                    feature_values = data[:, feature_idx]
                    n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                    # Loop over classes for true probabilities
                    for c in range(self.num_classes):
                        true_histogram, _ = np.histogram(
                            feature_values,
                            bins=bin_edges[feature_name],
                            weights=weights * labels_one_hot[:, c]
                        )
                        true_histograms[feature_name][:, c] += true_histogram

                    # Loop over classes for predicted probabilities
                    for c in range(self.num_classes):
                        pred_histogram, _ = np.histogram(
                            feature_values,
                            bins=bin_edges[feature_name],
                            weights=weights * predictions[:, c]
                        )
                        pred_histograms[feature_name][:, c] += pred_histogram

            total_loss += weighted_loss.numpy() * len(data)
            total_samples += len(data)
            i_batch += 1

            if max_batch > 0 and i_batch >= max_batch:
                break

        # Apply accumulated gradients after looping over the dataset
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
        epoch_loss = total_loss / total_samples
        print(f"Epoch loss: {epoch_loss:.4f}")

        if accumulate_histograms:
            return true_histograms, pred_histograms
        else:
            return None, None

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

        # Save the config name in a separate pickle file
        config_path = os.path.join(save_dir, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump((self.config_name, self.feature_means, self.feature_variances, self.weight_sums), f)

        # Manually create the 'checkpoint' metadata file
        with open(os.path.join(save_dir, 'checkpoint'), 'w') as f:
            f.write(f'model_checkpoint_path: "{checkpoint_path}"\n')

        print(f"Model checkpoint and config saved for epoch {epoch} in {save_dir}.")

    @classmethod
    def load(cls, save_dir):
        """
        Class method to load a saved TFMC instance from the latest checkpoint.
        Handles corrupted or missing config.pkl files gracefully.
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if not latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in directory: {save_dir}")

        # Load the config module name from the pickle file
        config_path = os.path.join(save_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                #config_name, feature_means, feature_variances, weight_sums = pickle.load(f)
                in_ = pickle.load(f)
                if len(in_)==4:
                    config_name, feature_means, feature_variances, weight_sums = in_
                else: #FIXME remove the 'else', it is for old trainings
                    weight_sums = {i:1./4. for i in range(4)} 
                    config_name, feature_means, feature_variances = in_ 
                    
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Failed to load config.pkl due to corruption: {e}")

        # Dynamically import the config module
        config = importlib.import_module(("ML." if config_name.startswith("configs.") else "") + config_name)

        # Create a new TFMC instance
        instance = cls(config=config)

        # Restore the model and optimizer state
        instance.checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Model and config loaded from {latest_checkpoint} with config {config_name}.")

        instance.feature_means     = feature_means
        instance.feature_variances = feature_variances
        instance.weight_sums       = weight_sums

        total = sum(instance.weight_sums.values())
        instance.class_weights = np.array([total/instance.weight_sums[i] for i in range(instance.num_classes)])

        return instance

    def plot_convergence_root(self, true_histograms, pred_histograms, epoch, output_path, feature_names):
        """
        Plot and save the convergence visualization for all features in one canvas using ROOT.

        Parameters:
        - true_histograms: dict, true class probabilities accumulated over bins.
        - pred_histograms: dict, predicted class probabilities accumulated over bins.
        - epoch: int, current epoch number.
        - output_path: str, directory to save the ROOT files.
        - feature_names: list of str, feature names for the x-axis.
        """
        ROOT.gStyle.SetOptStat(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()

        os.makedirs(output_path, exist_ok=True)

        num_features = len(feature_names)
        num_classes = len(self.classes)

        for normalized in [False, True]:
            if normalized:
                # Normalize histograms by the total truth sum
                for feature_name in data_structure.feature_names:
                    # Compute the total truth sum across all classes
                    total_truth_sum = true_histograms[feature_name].sum(axis=1, keepdims=True)

                    # Avoid division by zero for total truth sum
                    total_truth_sum = np.where(total_truth_sum == 0, 1, total_truth_sum)

                    # Normalize true histograms to ensure they sum to 1
                    true_histograms[feature_name] /= total_truth_sum

                    # Scale predicted histograms by the total truth sum (same normalization)
                    pred_histograms[feature_name] /= total_truth_sum

            # Calculate grid size, adding one pad for the legend
            total_pads = num_features + 1
            grid_size_x = int(ceil(sqrt(total_pads)))
            grid_size_y = int(ceil(total_pads/grid_size_x))
            canvas = ROOT.TCanvas("c_convergence", "Convergence Plot", 500*grid_size_x, 500*grid_size_y)
            canvas.Divide(grid_size_x, grid_size_y)

            colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kOrange, ROOT.kMagenta]  # Define a set of colors
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
                for c in range(num_classes):
                    max_y = max(
                        max_y,
                        true_histograms[feature_name][:, c].max(),
                        pred_histograms[feature_name][:, c].max(),
                    )

                # Fetch binning and axis title from plot_options
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]["binning"]
                x_axis_title = data_structure.plot_options[feature_name]["tex"]

                h_frame = ROOT.TH2F(
                    f"h_frame_{feature_name}",
                    f";{x_axis_title};Probability",
                    n_bins, x_min, x_max,
                    100, 0, 1.2 * max_y,
                )
                h_frame.GetYaxis().SetTitleOffset(1.3)
                h_frame.Draw()
                stuff.append(h_frame)

                # Loop through classes to create and style histograms
                for c, class_name in enumerate(self.classes):
                    # True probabilities (dashed)
                    h_true = ROOT.TH1F(
                        f"h_true_{feature_name}_{c}",
                        f"{feature_name} (true {class_name})",
                        n_bins, x_min, x_max,
                    )
                    for i, y in enumerate(true_histograms[feature_name][:, c]):
                        h_true.SetBinContent(i + 1, y)

                    h_true.SetLineColor(colors[c % len(colors)])
                    h_true.SetLineStyle(2)  # Dashed
                    h_true.SetLineWidth(2)
                    h_true.Draw("HIST SAME")
                    stuff.append(h_true)

                    # Predicted probabilities (solid)
                    h_pred = ROOT.TH1F(
                        f"h_pred_{feature_name}_{c}",
                        f"{feature_name} (pred {class_name})",
                        n_bins, x_min, x_max,
                    )
                    for i, y in enumerate(pred_histograms[feature_name][:, c]):
                        h_pred.SetBinContent(i + 1, y)

                    h_pred.SetLineColor(colors[c % len(colors)])
                    h_pred.SetLineStyle(1)  # Solid
                    h_pred.SetLineWidth(2)
                    h_pred.Draw("HIST SAME")
                    stuff.append(h_pred)

            # Legend in the last pad
            legend_pad_index = num_features + 1
            canvas.cd(legend_pad_index)

            legend = ROOT.TLegend(0.1, 0.1, 0.9, 0.9)
            legend.SetBorderSize(0)
            legend.SetShadowColor(0)

            # Create dummy histograms for legend
            dummy_true = []
            dummy_pred = []

            for c, class_name in enumerate(self.classes):
                # Dummy histogram for true probabilities
                hist_true = ROOT.TH1F(f"dummy_true_{c}", "", 1, 0, 1)
                hist_true.SetLineColor(colors[c % len(colors)])
                hist_true.SetLineStyle(2)  # Dashed
                hist_true.SetLineWidth(2)
                dummy_true.append(hist_true)

                # Dummy histogram for predicted probabilities
                hist_pred = ROOT.TH1F(f"dummy_pred_{c}", "", 1, 0, 1)
                hist_pred.SetLineColor(colors[c % len(colors)])
                hist_pred.SetLineStyle(1)  # Solid
                hist_pred.SetLineWidth(2)
                dummy_pred.append(hist_pred)

                # Add entries to the legend
                legend.AddEntry(hist_true, f"{class_name} (true)", "l")
                legend.AddEntry(hist_pred, f"{class_name} (pred)", "l")

            legend.Draw()
            stuff.extend(dummy_true + dummy_pred)

            tex = ROOT.TLatex()
            tex.SetNDC()
            tex.SetTextSize(0.07)
            tex.SetTextAlign(11) # align right

            lines = [ (0.3, 0.95, "Epoch =%5i"%epoch) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            # Save the canvas
            norm = "norm_" if normalized else ""
            output_file = os.path.join(output_path, f"{norm}epoch_{epoch:04d}.png")
            for fmt in ["png"]:  
                canvas.SaveAs(output_file.replace(".png", f".{fmt}"))

            print(f"Saved convergence plot for epoch {epoch} to {output_file}.")

