import xgboost as xgb
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure
import common.user as user
from tqdm import tqdm
import common.syncer
from math import ceil, sqrt

class XGBMC:
    def __init__(self, config=None, input_dim=None, classes=None, model_dir=None, num_boost_round=None):
        """
        Initialize the XGBoost multiclass classifier model.

        Parameters:
        - config: configuration object with hyperparameters.
        - input_dim: int, number of features in the input data.
        - classes: list of class labels.
        """
        if config is not None:
            self.config = config
            self.input_dim = config.input_dim
            self.classes = config.classes
            self.model_dir = config.model_dir
            self.num_boost_round = config.num_boost_round
        elif not ( input_dim is None or classes is None or model_dir is None or num_boost_round is None):
            self.config=None
            self.input_dim=input_dim
            self.classes=classes
            self.model_dir=model_dir
            self.num_boost_round=num_boost_round 
        else:
            raise Exception("Please provide a config.")

        self.num_classes = len(self.classes)
        self.model = None
        self.feature_means = getattr(config, 'feature_means', None)
        self.feature_variances = getattr(config, 'feature_variances', None)
        self.weight_sums = getattr(config, 'weight_sums', None)

    def load_training_data(self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader(selection=selection, selection_function=None, n_split=n_split)

    def train(self, max_batch=-1, every=-1, plot_directory=None):
        """
        Train the XGBoost classifier using the batched data loader.
        """
        if self.data_loader is None:
            raise ValueError("Data loader is not initialized. Call `load_training_data` first.")

        if not hasattr( self, "params" ):
            self.params = {
                'objective': 'multi:softprob',  # Multiclass classification with soft probabilities
                'num_class': self.num_classes,  # Number of classes
                'eta': self.config.learning_rate,  # Learning rate
                'max_depth': self.config.max_depth,  # Max depth of trees
                'subsample': self.config.subsample,  # Fraction of samples per tree
                'colsample_bytree': self.config.colsample_bytree,  # Fraction of features per tree
                'lambda': self.config.l2_reg,  # L2 regularization (alpha)
                'alpha': self.config.l1_reg,  # L1 regularization (lambda)
                'eval_metric': 'mlogloss',  # Cross-entropy loss
                'seed': self.config.seed,  # Random seed
            }

        for epoch in range(self.start_epoch, self.num_boost_round):

            accumulate_histograms = epoch%every==0 if every>0 else False

            if accumulate_histograms:
                # For histogram accumulation
                num_features = len(data_structure.feature_names)
                true_histograms = {}
                pred_histograms = {}
                bin_edges = {}

                # Initialize histograms based on plot_options
                for feature_name in data_structure.plot_options.keys():
                    n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
                    true_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                    pred_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                    bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)

            for i_batch, batch in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch}. Batches: ")):

                if max_batch>0 and i_batch>= max_batch: break

                data, weights, raw_labels = self.data_loader.split(batch)
                data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

                if self.weight_sums:
                    class_weights = np.array([
                        self.weight_sums[data_structure.label_encoding[label]] for label in self.classes
                    ])
                    total = sum(class_weights)
                    class_weights = np.array([total / class_weights[i] for i in range(len(class_weights))])
                    weights *= class_weights[raw_labels.astype('int')]

                dtrain = xgb.DMatrix(data_norm, label=raw_labels, weight=weights)

                if self.model is None:
                    self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=1, xgb_model=None)
                else:
                    self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=1, xgb_model=self.model)

                # Accumulate histograms if requested
                if accumulate_histograms:
                    for feature_idx, feature_name in enumerate(data_structure.feature_names):
                        feature_values = data[:, feature_idx]
                        n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                        # Loop over classes for true probabilities
                        labels_one_hot = np.eye(self.num_classes)[raw_labels.astype('int')]
                        for c in range(self.num_classes):
                            true_histogram, _ = np.histogram(
                                feature_values,
                                bins=bin_edges[feature_name],
                                weights=weights * labels_one_hot[:, c]
                            )
                            true_histograms[feature_name][:, c] += true_histogram
                            #print ("true", feature_name, c, (weights * labels_one_hot[:, c]).mean() )

                        # Loop over classes for predicted probabilities
                        predictions = self.model.predict(xgb.DMatrix(data_norm))  # Probabilities for each class
                        for c in range(self.num_classes):
                            pred_histogram, _ = np.histogram(
                                feature_values,
                                bins=bin_edges[feature_name],
                                weights=weights * predictions[:, c]
                            )
                            pred_histograms[feature_name][:, c] += pred_histogram

            if accumulate_histograms:
                self.plot_convergence_root(true_histograms, pred_histograms, epoch, plot_directory=plot_directory, feature_names=data_structure.feature_names)
                common.syncer.sync()

            self.save(epoch=epoch + 1)

    def predict(self, data, ic_scaling=True):

        # apply scaler
        data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

        dtest = xgb.DMatrix(data_norm)
        class_probs = self.model.predict(dtest)  # Probabilities for each class

        # put back the inclusive xsec
        if ic_scaling:
            class_weights = np.array([self.weight_sums[data_structure.label_encoding[label]] for label in self.classes])
            total = sum(class_weights)
            class_weights = np.array([total / class_weights[i] for i in range(len(class_weights))])
            return class_probs/class_weights # DCR
        else:
            return class_probs

    def save(self, epoch):
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.json")
        metadata_path = os.path.join(self.model_dir, f"model_metadata_{epoch:04d}.pkl")

        os.makedirs(self.model_dir, exist_ok=True)

        if self.model:
            self.model.save_model(model_path)
            metadata = {
                'feature_means': self.feature_means,
                'feature_variances': self.feature_variances,
                'weight_sums': self.weight_sums,
                'epoch': epoch,
                'input_dim': self.input_dim,
                'classes': self.classes,
                'model_dir': self.model_dir,
                'num_boost_round': self.num_boost_round,
                'params':self.params,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            #print(f"Model and metadata saved for epoch {epoch}")
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
        instance = cls(config=None, input_dim=metadata['input_dim'], classes=metadata['classes'],model_dir=metadata['model_dir'],num_boost_round=metadata['num_boost_round'])
        instance.model = xgb.Booster()
        instance.model.load_model(model_path)
        instance.model_dir = model_dir
        instance.feature_means = metadata['feature_means']
        instance.feature_variances = metadata['feature_variances']
        instance.weight_sums = metadata['weight_sums']
        instance.params = metadata['params']
        print(f"Model and metadata loaded from {model_path}, epoch {epoch}")
        if return_epoch:
            return instance, epoch
        else:
            return instance

    def plot_convergence_root(self, true_histograms, pred_histograms, epoch, plot_directory, feature_names):
        """
        Plot and save the convergence visualization for all features in one canvas using ROOT.

        Parameters:
        - true_histograms: dict, true class probabilities accumulated over bins.
        - pred_histograms: dict, predicted class probabilities accumulated over bins.
        - epoch: int, current epoch number.
        - plot_directory: str, directory to save the ROOT files.
        - feature_names: list of str, feature names for the x-axis.
        """
        import ROOT
        ROOT.gStyle.SetOptStat(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()

        os.makedirs(plot_directory, exist_ok=True)

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
            output_file = os.path.join(plot_directory, f"{norm}epoch_{epoch:04d}.png")
            for fmt in ["png"]:  
                canvas.SaveAs(output_file.replace(".png", f".{fmt}"))

            print(f"Saved convergence plot for epoch {epoch} to {output_file}.")
