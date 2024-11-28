import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from math import ceil, sqrt

class MulticlassClassifier:
    def __init__(self, input_dim, num_classes):
        """
        Initialize the multiclass classifier model.
        
        Parameters:
        - input_dim: int, number of features in the input data.
        - num_classes: int, number of output classes.
        """
        self.model = self._build_model(input_dim, num_classes)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.metrics = tf.keras.metrics.CategoricalAccuracy()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    def _build_model(self, input_dim, num_classes):
        """Build a simple neural network for classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model 

    def train_one_epoch(self, data_loader, class_labels, max_batch=-1):
        """
        Train the model for one epoch using the data loader.
        
        Parameters:
        - data_loader: H5DataLoader, for loading batches of data.
        - class_labels: list of str, class names in the dataset.
        """
        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        total_loss = 0.0
        total_samples = 0
        i_batch = 0
        for batch in data_loader:
            print(f"Batch {i_batch}")
            data = batch['data']
            weights = batch['weights']
            raw_labels = batch['detailed_labels']
            
            # Convert raw labels to one-hot encoded format
            labels = np.array([class_labels.index(label.decode('utf-8')) for label in raw_labels])
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(class_labels))
            
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
    
    def evaluate(self, data_loader, class_labels, max_batch=-1):
        """
        Evaluate the model on the data loader.
        
        Parameters:
        - data_loader: H5DataLoader, for loading batches of data.
        - class_labels: list of str, class names in the dataset.
        """
        total_samples = 0
        self.metrics.reset_states()

        i_batch = 0
        for batch in data_loader:
            data = batch['data']
            raw_labels = batch['detailed_labels']
            
            # Convert raw labels to one-hot encoded format
            labels = np.array([class_labels.index(label.decode('utf-8')) for label in raw_labels])
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(class_labels))
            
            predictions = self.model(data, training=False)
            self.metrics.update_state(labels_one_hot, predictions)
            total_samples += len(data)
            i_batch+=1
            if max_batch>0 and i_batch>=max_batch:
                break
        
        print(f"Validation accuracy: {self.metrics.result().numpy():.4f}")

    def save(self, save_dir, epoch):
        """
        Save the model and optimizer state to a file.

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

        print(f"Model checkpoint saved for epoch {epoch} in {checkpoint_path}.")

    def load(self, save_dir, checkpoint=None):
        """
        Load the model and optimizer state from a checkpoint.

        Parameters:
        - save_dir: str, directory where checkpoints are stored (e.g., 'models/test').
        - checkpoint: int or None, specific epoch number to load (e.g., 5).
                      If None, the latest checkpoint will be loaded.
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        if checkpoint is None:
            # Find the latest checkpoint using TensorFlow's metadata
            latest_checkpoint = tf.train.latest_checkpoint(save_dir)
            if not latest_checkpoint:
                raise FileNotFoundError(f"No checkpoint found in directory: {save_dir}")
            checkpoint_path = latest_checkpoint
        else:
            # Use the specified epoch as the checkpoint filename
            checkpoint_path = os.path.join(save_dir, str(checkpoint))
            if not os.path.exists(f"{checkpoint_path}.index"):  # Checkpoint files include '.index'
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.checkpoint.restore(checkpoint_path).expect_partial()
        print(f"Model checkpoint loaded from {checkpoint_path}.")

    def accumulate_histograms(self, data_loader, class_labels, n_bins=30, max_batch=-1):
        """
        Accumulate histograms of true and predicted class probabilities for visualization.

        Parameters:
        - data_loader: H5DataLoader, for loading batches of data.
        - class_labels: list of str, class names in the dataset.
        - n_bins: int, number of bins for histograms (default: 30).
        - max_batch: int, maximum number of batches to process (default: -1, process all).
        """
        n_features = self.model.input_shape[1]
        n_classes = len(class_labels)
        bin_edges = []
        true_histograms = {k: np.zeros((n_bins, n_classes)) for k in range(n_features)}
        pred_histograms = {k: np.zeros((n_bins, n_classes)) for k in range(n_features)}

        i_batch = 0
        for batch in data_loader:
            data = batch['data']
            weights = batch['weights']
            raw_labels = batch['detailed_labels']
            predictions = self.model(data, training=False).numpy()

            # Convert raw labels to one-hot encoded format
            labels = np.array([class_labels.index(label.decode('utf-8')) for label in raw_labels])
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=n_classes)

            for k in range(n_features):
                feature_values = data[:, k]

                # Define bin edges only once
                if i_batch == 0:
                    bin_edges.append(np.linspace(feature_values.min(), feature_values.max(), n_bins + 1))

                # Accumulate true and predicted probabilities in bins
                for b in range(n_bins):
                    in_bin = (feature_values >= bin_edges[k][b]) & (feature_values < bin_edges[k][b + 1])
                    bin_weights = weights[in_bin]

                    # True class probabilities
                    if bin_weights.sum() > 0:
                        true_histograms[k][b, :] += np.sum(bin_weights[:, None] * labels_one_hot[in_bin], axis=0)

                    # Predicted class probabilities
                    if bin_weights.sum() > 0:
                        pred_histograms[k][b, :] += np.sum(bin_weights[:, None] * predictions[in_bin], axis=0)

            i_batch += 1
            if max_batch > 0 and i_batch >= max_batch:
                break

        # Normalize histograms
        for k in range(n_features):
            true_sums = true_histograms[k].sum(axis=1, keepdims=True)
            pred_sums = pred_histograms[k].sum(axis=1, keepdims=True)
            true_histograms[k] /= np.where(true_sums == 0, 1, true_sums)
            pred_histograms[k] /= np.where(pred_sums == 0, 1, pred_sums)

        return true_histograms, pred_histograms, bin_edges

    def plot_convergence(self, true_histograms, pred_histograms, bin_edges, epoch, output_path, class_labels, feature_names):
        """
        Plot and save the convergence visualization for each feature.

        Parameters:
        - true_histograms: dict, true class probabilities accumulated over bins.
        - pred_histograms: dict, predicted class probabilities accumulated over bins.
        - bin_edges: list, bin edges for each feature.
        - epoch: int, current epoch number.
        - output_path: str, directory to save the PNG files.
        - class_labels: list of str, class names.
        - feature_names: list of str, feature names for the x-axis.
        """
        n_features = len(true_histograms)
        n_classes = len(class_labels)

        # Calculate grid size dynamically to fit all features
        grid_size = ceil(sqrt(n_features))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        colors = plt.cm.tab10(np.arange(n_classes))  # Use tab10 colormap for distinct colors

        for k in range(n_features):
            ax = axes[k]
            bin_centers = 0.5 * (bin_edges[k][:-1] + bin_edges[k][1:])

            for c, class_name in enumerate(class_labels):
                # Dashed lines for true probabilities
                ax.plot(
                    bin_centers,
                    true_histograms[k][:, c],
                    linestyle="--",
                    color=colors[c],
                    label=f"{class_name} (true)" if k == 0 else "",
                )

                # Solid lines for predicted probabilities
                ax.plot(
                    bin_centers,
                    pred_histograms[k][:, c],
                    linestyle="-",
                    color=colors[c],
                    label=f"{class_name} (pred)" if k == 0 else "",
                )

            ax.set_title(feature_names[k])
            ax.set_xlabel(feature_names[k])  # Use feature name for x-axis
            ax.set_ylabel("Probability")
            ax.grid(True)

        # Hide unused subplots
        for ax in axes[n_features:]:
            ax.axis("off")

        # Add legend to the figure
        handles = [
            plt.Line2D([0], [0], color=colors[c], linestyle="--", label=f"{class_name} (true)")
            for c, class_name in enumerate(class_labels)
        ] + [
            plt.Line2D([0], [0], color=colors[c], linestyle="-", label=f"{class_name} (pred)")
            for c, class_name in enumerate(class_labels)
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=n_classes, frameon=False)

        output_file = os.path.join(output_path, f"epoch_{epoch}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close(fig)
        print(f"Saved convergence plot for epoch {epoch} to {output_file}.")

