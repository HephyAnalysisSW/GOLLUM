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

import common.data_structure as data_structure

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal, Constant

#class CustomDense(Dense):
#    """ This class makes the initial outputs close to 1/num_classes
#    """
#    def __init__(self, units, **kwargs):
#        super().__init__(units, **kwargs)
#
#    def build(self, input_shape):
#        super().build(input_shape)
#
#        # Ensure weights are initialized normally (default in most cases)
#        initializer = RandomNormal(mean=0.0, stddev=0.05)
#        self.kernel_initializer = initializer
#
#        # Bias initialization
#        num_classes = self.units
#        target_logits = tf.math.log(tf.constant(1.0 / num_classes, dtype=tf.float32))
#        bias_perturbation = tf.random.normal(
#            shape=(num_classes,),
#            mean=0.0,
#            stddev=0.05,
#            dtype=tf.float32
#        )
#        self.bias_initializer = Constant(target_logits + bias_perturbation)


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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.metrics = tf.keras.metrics.CategoricalAccuracy()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        if hasattr( config, "weight_sums"):
            self.weight_sums = config.weight_sums
        else:
            self.weight_sums = {i:1 for i in range(self.num_classes)}


        if hasattr( config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            self.feature_means     = {i:0 for i in range(len(data_structure.feature_names))}
            self.feature_variances = {i:1 for i in range(len(data_structure.feature_names))}

        # Scale cross sections to the same integral
        total = sum(self.weight_sums.values())
        self.scales = np.array([total/self.weight_sums[i] for i in range(self.num_classes)])
        
        print("Will scale with these factors: "+" ".join( ["%s: %3.2f"%( self.classes[i], self.scales[i]) for i in range( self.num_classes)]) )

    def _build_model(self):
        """Build a simple neural network for classification with batch normalization."""
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        # Hidden layers with batch normalization
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=None))  # No activation yet
            #model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))  # Apply activation after normalization

        # Output layer
        #model.add(CustomDense(self.num_classes, activation='softmax'))
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def predict(self, data, ic_scaling=True):   
        res =  self.model((data - self.feature_means) / np.sqrt(self.feature_variances), training=False).numpy()
        # put back the inclusive xsec
        if ic_scaling:
            return res/self.scales # DCR
        else:
            return res             # LR

    def load_training_data( self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader( selection=selection, selection_function=None, n_split=n_split)

    def train_one_epoch(self, max_batch=-1):
        """
        Train the model for one epoch using the data loader.
        
        Parameters:
        """
        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        total_loss = 0.0
        total_samples = 0
        i_batch = 0
        for batch in self.data_loader:
            print(f"Batch {i_batch}")
            data, weights, raw_labels = self.data_loader.split(batch)

            # Scaler
            data = (data - self.feature_means) / np.sqrt(self.feature_variances)

            # reweighting
            if self.reweighting:
                weights = weights * self.scales[raw_labels.astype('int')]

            # Convert raw labels to one-hot encoded format
            labels_one_hot = tf.keras.utils.to_categorical(raw_labels, num_classes=self.num_classes)
 
            with tf.GradientTape() as tape:
                predictions = self.model(data, training=True)
                loss = self.loss_fn(labels_one_hot, predictions)
                weighted_loss = tf.reduce_mean(loss * weights)
            
            gradients = tape.gradient(weighted_loss, self.model.trainable_variables)
            accumulated_gradients = [
                acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
            ]
            
            total_loss += weighted_loss.numpy() * len(data)
            total_samples += len(data)
            i_batch+=1
            if max_batch>0 and i_batch>=max_batch:
                break

        # Apply accumulated gradients after looping over the dataset
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
        epoch_loss = total_loss / total_samples
        print(f"Epoch loss: {epoch_loss:.4f}")
    
    def evaluate(self, max_batch=-1):
        """
        Evaluate the model on the data loader.
        
        Parameters:
        """
        total_samples = 0
        self.metrics.reset_states()

        i_batch = 0
        for batch in self.data_loader:
            data, weights, raw_labels = self.data_loader.split(batch)
            
            # Convert raw labels to one-hot encoded format
            labels_one_hot = tf.keras.utils.to_categorical(raw_labels, num_classes=self.num_classes)
            
            predictions = self.model(data, training=False)
            self.metrics.update_state(labels_one_hot, predictions)
            total_samples += len(data)
            i_batch+=1
            if max_batch>0 and i_batch>=max_batch:
                break
        
        print(f"Validation accuracy: {self.metrics.result().numpy():.4f}")

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
            pickle.dump(self.config_name, f)

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
                config_name = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Failed to load config.pkl due to corruption: {e}")

        # Dynamically import the config module
        config = importlib.import_module(config_name)

        # Create a new TFMC instance
        instance = cls(config=config)

        # Restore the model and optimizer state
        instance.checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Model and config loaded from {latest_checkpoint} with config {config_name}.")

        return instance

    def accumulate_histograms(self, max_batch=-1):
        """
        Accumulate histograms of true and predicted class probabilities for visualization.

        Parameters:
        - max_batch: int, maximum number of batches to process (default: -1, process all).

        Returns:
        - true_histograms: dict, true class probabilities accumulated over bins.
        - pred_histograms: dict, predicted class probabilities accumulated over bins.
        - bin_edges: dict, bin edges for each feature.
        """

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

        i_batch = 0
        for batch in self.data_loader:
            data, weights, raw_labels = self.data_loader.split(batch)
            predictions = self.predict(data, ic_scaling=False)
            # Apply reweighting if enabled
            if self.reweighting:
                weights = weights * self.scales[raw_labels.astype('int')]

            # Convert raw labels to one-hot encoded format
            labels_one_hot = tf.keras.utils.to_categorical(raw_labels, num_classes=self.num_classes)

            # Loop through each feature
            for feature_idx, feature_name in enumerate(data_structure.feature_names):
                feature_values = data[:, feature_idx]
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                # Accumulate true and predicted probabilities in bins
                for b in range(n_bins):
                    in_bin = (feature_values >= bin_edges[feature_name][b]) & (
                        feature_values < bin_edges[feature_name][b + 1]
                    )
                    bin_weights = weights[in_bin]

                    # True class probabilities
                    if bin_weights.sum() > 0:
                        true_histograms[feature_name][b, :] += np.sum(
                            bin_weights[:, None] * labels_one_hot[in_bin], axis=0
                        )

                    # Predicted class probabilities
                    if bin_weights.sum() > 0:
                        pred_histograms[feature_name][b, :] += np.sum(
                            bin_weights[:, None] * predictions[in_bin], axis=0
                        )

            i_batch += 1
            if max_batch > 0 and i_batch >= max_batch:
                break

        return true_histograms, pred_histograms

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
                # Normalize histograms
                for feature_name in data_structure.feature_names:
                    true_sums = true_histograms[feature_name].sum(axis=1, keepdims=True)
                    pred_sums = pred_histograms[feature_name].sum(axis=1, keepdims=True)
                    true_histograms[feature_name] /= np.where(true_sums == 0, 1, true_sums)
                    pred_histograms[feature_name] /= np.where(pred_sums == 0, 1, pred_sums)

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
                
                pad.SetLogy(not normalized)

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

