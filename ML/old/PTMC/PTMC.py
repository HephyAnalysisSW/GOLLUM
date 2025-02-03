import torch
print("Num GPUs Available: ", torch.cuda.device_count())
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
from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import common.data_structure as data_structure

from torch.nn import Module, Linear, Dropout
from torch.optim import Adam
from torch.utils.data import DataLoader

class PhaseoutScheduler:

    def __init__(self, initial_lr, n_epochs, n_epochs_phaseout):
        self.initial_lr = initial_lr
        self.n_epochs = n_epochs
        self.n_epochs_phaseout = n_epochs_phaseout

    def __call__(self, epoch):
        if self.n_epochs_phaseout <= 0:
            return self.initial_lr
        elif epoch < self.n_epochs - self.n_epochs_phaseout:
            return self.initial_lr
        else:
            decay_start_epoch = self.n_epochs - self.n_epochs_phaseout
            decay_rate = self.initial_lr / self.n_epochs_phaseout
            return self.initial_lr - decay_rate * (epoch - decay_start_epoch)

class PTMC(Module):
    def __init__(self, config=None, input_dim=None, classes=None, hidden_layers=None, reweighting=True):
        super(PTMC, self).__init__()

        self.reweighting = reweighting

        if config is not None:
            self.config = config
            self.config_name = config.__name__
            self.input_dim = config.input_dim
            self.classes = config.classes
            self.hidden_layers = config.hidden_layers
        elif input_dim is not None and classes is not None and hidden_layers is not None:
            self.config = None
            self.config_name = None
            self.input_dim = input_dim
            self.classes = classes
            self.hidden_layers = hidden_layers
        else:
            raise Exception("Please provide either a config or all other parameters (input_dim, classes, hidden_layers).")

        self.num_classes = len(self.classes)

        self.model = self._build_model()

        self.lr_schedule = PhaseoutScheduler(
            initial_lr=config.learning_rate,
            n_epochs=config.n_epochs,
            n_epochs_phaseout=config.__dict__.get("n_epochs_phaseout", 0),
        )

        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        if hasattr( config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            self.feature_means     = np.array([0 for i in range(len(data_structure.feature_names))])
            self.feature_variances = np.array([1 for i in range(len(data_structure.feature_names))])

        if hasattr( config, "weight_sums"):
            self.weight_sums = config.weight_sums
        else:
            self.weight_sums = {i:1./self.num_classes for i in range(self.num_classes)}

        # Scale cross sections to the same integral
        self.class_weights = [ self.weight_sums[data_structure.label_encoding[label]] for label in self.config.classes ]
        total = sum(self.class_weights)
        self.class_weights = np.array([total/self.class_weights[i] for i in range(len(self.class_weights))])

        print("Will scale with these factors: "+" ".join( ["%s: %3.2f"%( self.classes[i], self.class_weights[i]) for i in range( self.num_classes)]) )

    def _build_model(self):
        layers = []
        input_size = self.input_dim

        for units in self.hidden_layers:
            layers.append(Linear(input_size, units))
            layers.append(torch.nn.ReLU())
            if self.config.__dict__.get("dropout_rate", 0) > 0:
                layers.append(Dropout(self.config.__dict__["dropout_rate"]))
            input_size = units

        layers.append(Linear(input_size, self.num_classes))
        layers.append(torch.nn.Softmax(dim=1))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self, data, ic_scaling=True):
        # Normalization of inputs
        data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)

        # preprocess features, if needed
        if hasattr( self.config, "preprocessor"):
            data_norm = self.config.preprocessor( data, data_norm)

        with torch.no_grad():
            res = self.model(torch.tensor(data_norm, dtype=torch.float32)).numpy()
        if ic_scaling:
            return res / self.class_weights
        else:
            return res

    def load_training_data(self, datasets, selection, n_split=10):
        self.data_loader = datasets.get_data_loader(selection=selection, selection_function=None, n_split=n_split)

    def train_one_epoch(self, max_batch=-1, accumulate_histograms=False):
        if accumulate_histograms:
            true_histograms = {}
            pred_histograms = {}
            bin_edges = {}

            for feature_name in data_structure.plot_options.keys():
                n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']
                true_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                pred_histograms[feature_name] = np.zeros((n_bins, self.num_classes))
                bin_edges[feature_name] = np.linspace(x_min, x_max, n_bins + 1)

        self.optimizer.zero_grad()
        total_loss = 0.0
        total_samples = 0
        selected_indices = [data_structure.label_encoding[label] for label in self.config.classes]

        for i_batch, batch in enumerate(tqdm(self.data_loader, desc="Processing Batches")):
            data, weights, raw_labels = self.data_loader.split(batch)

            mask = np.isin(raw_labels, selected_indices)
            data = data[mask]
            weights = weights[mask]
            raw_labels = raw_labels[mask]

            data_norm = (data - self.feature_means) / np.sqrt(self.feature_variances)
            data_norm = torch.tensor(data_norm, dtype=torch.float32)

            if hasattr(self.config, "preprocessor"):
                data_norm = torch.tensor(self.config.preprocessor(data, data_norm), dtype=torch.float32)

            if self.reweighting:
                weights = weights * self.class_weights[raw_labels.astype('int')]

            remapped_labels = np.array([selected_indices.index(label) for label in raw_labels])
            labels_one_hot = torch.nn.functional.one_hot(torch.tensor(remapped_labels), num_classes=self.num_classes).float()

            predictions = self.model(data_norm)
            loss = self.loss_fn(predictions, labels_one_hot).mean()
            weighted_loss = (loss * torch.tensor(weights, dtype=torch.float32)).mean()

            weighted_loss.backward()
            total_loss += weighted_loss.item() * len(data)
            total_samples += len(data)

            # Accumulate histograms if requested
            if accumulate_histograms:
                with torch.no_grad():
                    for feature_idx, feature_name in enumerate(data_structure.feature_names):
                        feature_values = data[:, feature_idx]
                        n_bins, x_min, x_max = data_structure.plot_options[feature_name]['binning']

                        # Loop over classes for true probabilities
                        for c in range(self.num_classes):
                            true_histogram, _ = np.histogram(
                                feature_values,
                                bins=bin_edges[feature_name],
                                weights=weights * labels_one_hot[:, c].numpy()
                            )
                            true_histograms[feature_name][:, c] += true_histogram
                            #print ("true", feature_name, c, (weights * labels_one_hot[:, c].numpy()).mean() )

                        # Loop over classes for predicted probabilities
                        for c in range(self.num_classes):
                            pred_histogram, _ = np.histogram(
                                feature_values,
                                bins=bin_edges[feature_name],
                                weights=weights * predictions[:, c].numpy()
                            )
                            pred_histograms[feature_name][:, c] += pred_histogram
                            #print ("pred", feature_name, c, (weights * predictions[:, c].numpy()).mean() )
                #assert False, ""

            if max_batch > 0 and i_batch + 1 >= max_batch:
                break

        self.optimizer.step()
        epoch_loss = total_loss / total_samples
        print(f"Epoch loss: {epoch_loss:.4f}")

        if accumulate_histograms:
            return true_histograms, pred_histograms
        else:
            return None, None

    def save(self, save_dir, epoch):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config_name': self.config_name,
            'feature_means': self.feature_means,
            'feature_variances': self.feature_variances,
            'weight_sums': self.weight_sums
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    @classmethod
    def load(cls, save_dir):
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {save_dir}")
        latest_checkpoint = os.path.join(save_dir, sorted(checkpoints)[-1])
        checkpoint = torch.load(latest_checkpoint)

        config = importlib.import_module(checkpoint['config_name'])
        instance = cls(config=config)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        instance.feature_means = checkpoint['feature_means']
        instance.feature_variances = checkpoint['feature_variances']
        instance.weight_sums = checkpoint['weight_sums']

        total = sum(instance.weight_sums.values())
        instance.class_weights = np.array([total/instance.weight_sums[i] for i in range(instance.num_classes)])

        print(f"Model loaded from {latest_checkpoint}")
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

