import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
import operator
import functools

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
        self.VkA  = np.zeros( [len(self.base_points), len(self.combinations) ], dtype='float64')
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
        # Compute matrix Mkk from non-nominal base_points
        self.MkA  = np.dot(self._VKA, self.CInv).transpose()
        self.Mkkp = np.dot(self._VKA, self.MkA )

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        #self.loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        #self.metrics = tf.keras.metrics.CategoricalAccuracy()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        #if hasattr( config, "weight_sums"):
        #    self.weight_sums = config.weight_sums
        #else:
        #    self.weight_sums = {i:1 for i in range(self.num_classes)}

        if hasattr( config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            self.feature_means     = {i:0 for i in range(len(data_structure.feature_names))}
            self.feature_variances = {i:1 for i in range(len(data_structure.feature_names))}

        ## Scale cross sections to the same integral
        #total = sum(self.weight_sums.values())
        #self.scales = np.array([total/self.weight_sums[i] for i in range(self.num_classes)])

        #print("Will scale with these factors: "+" ".join( ["%s: %3.2f"%( self.classes[i], self.scales[i]) for i in range( self.num_classes)]) )

    def _build_model(self):
        """Build a simple neural network for classification with batch normalization."""
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        # Hidden layers with batch normalization
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=None))  # No activation yet
            model.add(tf.keras.layers.Activation('relu'))  # Apply activation after normalization

        # Output layer
        model.add(Dense(self.num_outputs, activation=None))

        return model

    def load_training_data( self, datasets, selection, n_split=10):
        self.training_data = {}
        for base_point in self.base_points:
            base_point = tuple(base_point)
            values = self.config.get_alpha(base_point)
            data_loader = datasets.get_data_loader( selection=selection, values=values, selection_function=None, n_split=n_split)
            print ("PNN training data: Base point nu = %r, alpha = %r, file = %s"%( base_point, values, data_loader.file_path))
            self.training_data[base_point] = data_loader

    def train_one_epoch(self, max_batch=-1, accumulate_histograms=False):
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
            # For histogram accumulation
            true_histograms = {}
            pred_histograms = {}
            bin_edges = {}

            # Initialize histograms based on plot_options
            for feature_name in data_structure.plot_options.keys():
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
                true_histograms[feature_name] = np.zeros((n_bins, len(self.base_points)))
                pred_histograms[feature_name] = np.zeros((n_bins, len(self.base_points)))
                bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)

        total_loss = 0.0
        i_batch = 0

        for batch in self.training_data[self.nominal_base_point_key]:
            print(f"Batch {i_batch}")
            features, weights_nominal, _ = self.training_data[nominal_base_point_index].split(batch)

            # Normalize features
            features_norm = (features - self.feature_means) / np.sqrt(self.feature_variances)

            # Nominal predictions
            DeltaA_nominal = self.model(features_norm, training=True)

            for i_base_point, base_point in enumerate(self.base_points):
                if i_base_point == self.nominal_base_point_index:
                    continue

                # Retrieve features and weights for the current base point
                features_nu, weights_nu, _ = self.data_loader.split(self.training_data[tuple(base_point)])
                features_nu_norm = (features_nu - self.feature_means) / np.sqrt(self.feature_variances)
                DeltaA = self.model(features_nu_norm, training=True)

                # Compute weighted losses
                loss_0 = tf.reduce_sum(
                    weights_nominal * tf.math.softplus(tf.linalg.matvec(DeltaA_nominal, self.VkA[i_base_point]))
                )
                loss_nu = tf.reduce_sum(
                    weights_nu * tf.math.softplus(-tf.linalg.matvec(DeltaA, self.VkA[i_base_point]))
                )
                loss = loss_0 + loss_nu
                loss -= (tf.reduce_sum(weights_nominal) + tf.reduce_sum(weights_nu)) * np.log(2.0)

                # Accumulate loss
                total_loss += loss.numpy()

                if accumulate_histograms:
                    for feature_idx, feature_name in enumerate(data_structure.feature_names):
                        feature_values = features[:, feature_idx]
                        n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                        # Accumulate true and predicted probabilities in bins
                        for b in range(n_bins):
                            in_bin = (feature_values >= bin_edges[feature_name][b]) & (
                                feature_values < bin_edges[feature_name][b + 1]
                            )
                            bin_weights_nominal = weights_nominal[in_bin]
                            bin_weights_nu = weights_nu[in_bin]

                            # True probabilities
                            if bin_weights_nominal.sum() > 0:
                                true_histograms[feature_name][b, i_base_point] += bin_weights_nominal.sum()
                            if bin_weights_nu.sum() > 0:
                                true_histograms[feature_name][b, i_base_point] += bin_weights_nu.sum()

                            # Predicted probabilities
                            if bin_weights_nominal.sum() > 0:
                                pred_histograms[feature_name][b, i_base_point] += np.sum(
                                    bin_weights_nominal * np.exp(tf.linalg.matvec(DeltaA_nominal, self.VkA[i_base_point])).numpy()
                                )
                            if bin_weights_nu.sum() > 0:
                                pred_histograms[feature_name][b, i_base_point] += np.sum(
                                    bin_weights_nu * np.exp(tf.linalg.matvec(DeltaA, self.VkA[i_base_point])).numpy()
                                )

            i_batch += 1
            if max_batch > 0 and i_batch >= max_batch:
                break

        # Apply gradients
        self.optimizer.apply_gradients(zip(self.model.trainable_variables, gradients))

        print(f"Epoch loss: {total_loss:.4f}")

        if accumulate_histograms:
            # Normalize histograms
            for feature_name in true_histograms.keys():
                true_sums = true_histograms[feature_name].sum(axis=1, keepdims=True)
                pred_sums = pred_histograms[feature_name].sum(axis=1, keepdims=True)
                true_histograms[feature_name] /= np.where(true_sums == 0, 1, true_sums)
                pred_histograms[feature_name] /= np.where(pred_sums == 0, 1, pred_sums)

            return true_histograms, pred_histograms
        else:
            return None, None

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
                # Normalize histograms
                for feature_name in feature_names:
                    true_sums = true_histograms[feature_name].sum(axis=1, keepdims=True)
                    pred_sums = pred_histograms[feature_name].sum(axis=1, keepdims=True)
                    true_histograms[feature_name] /= np.where(true_sums == 0, 1, true_sums)
                    pred_histograms[feature_name] /= np.where(pred_sums == 0, 1, pred_sums)

            # Calculate grid size, adding one pad for the legend
            total_pads = num_features + 1
            grid_size_x = int(ceil(sqrt(total_pads)))
            grid_size_y = int(ceil(total_pads / grid_size_x))
            canvas = ROOT.TCanvas("c_convergence", "Convergence Plot", 500 * grid_size_x, 500 * grid_size_y)
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
                for i_base_point in range(num_base_points):
                    max_y = max(
                        max_y,
                        true_histograms[feature_name][:, i_base_point].max(),
                        pred_histograms[feature_name][:, i_base_point].max(),
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

            print(f"Saved convergence plot for epoch {epoch} to {output_file}.")

