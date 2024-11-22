
from Multiclassifier import MulticlassClassifier
from data_loader.data_loader_2 import data_loader, feature_names

# Initialize model
input_dim = 28  # Number of features
num_classes = 4
model = MulticlassClassifier(input_dim, num_classes)

# Training Loop
epochs = 10
save_path = 'model_checkpoint.npz'

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train_one_epoch(data_loader, class_labels)
    model.save(save_path)  # Save model after each epoch

    # Evaluate on the same data for simplicity (use a validation set in practice)
    data_loader = H5DataLoader(file_path, datasets, n_split=n_split)  # Reload for evaluation
    model.evaluate(data_loader, class_labels)

# Load the saved model for further use
model.load(save_path)

