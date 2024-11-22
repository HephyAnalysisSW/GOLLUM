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
    
    def _build_model(self, input_dim, num_classes):
        """Build a simple neural network for classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def train_one_epoch(self, data_loader, class_labels):
        """
        Train the model for one epoch using the data loader.
        
        Parameters:
        - data_loader: H5DataLoader, for loading batches of data.
        - class_labels: list of str, class names in the dataset.
        """
        accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
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

        # Apply accumulated gradients after looping over the dataset
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
        epoch_loss = total_loss / total_samples
        print(f"Epoch loss: {epoch_loss:.4f}")
    
    def evaluate(self, data_loader, class_labels):
        """
        Evaluate the model on the data loader.
        
        Parameters:
        - data_loader: H5DataLoader, for loading batches of data.
        - class_labels: list of str, class names in the dataset.
        """
        total_samples = 0
        self.metrics.reset_states()

        for batch in data_loader:
            data = batch['data']
            raw_labels = batch['detailed_labels']
            
            # Convert raw labels to one-hot encoded format
            labels = np.array([class_labels.index(label.decode('utf-8')) for label in raw_labels])
            labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(class_labels))
            
            predictions = self.model(data, training=False)
            self.metrics.update_state(labels_one_hot, predictions)
            total_samples += len(data)
        
        print(f"Validation accuracy: {self.metrics.result().numpy():.4f}")
    
    def save(self, filename):
        """
        Save the model's weights and optimizer state to a file.
        
        Parameters:
        - filename: str, file path to save the model.
        """
        checkpoint = {
            'model_weights': self.model.get_weights(),
            'optimizer_state': self.optimizer.get_weights()
        }
        np.savez_compressed(filename, **checkpoint)
        print(f"Model saved to {filename}.")
    
    def load(self, filename):
        """
        Load the model's weights and optimizer state from a file.
        
        Parameters:
        - filename: str, file path to load the model from.
        """
        checkpoint = np.load(filename)
        self.model.set_weights(checkpoint['model_weights'])
        self.optimizer.set_weights(checkpoint['optimizer_state'])
        print(f"Model loaded from {filename}.")

