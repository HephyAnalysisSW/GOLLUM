import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from Multiclassifier import MulticlassClassifier

from data import get_data_loader, feature_names
import common.user as user
import common.syncer

data_loader = get_data_loader( n_split=1000 )

# Initialize model
input_dim = 28  # Number of features
num_classes = 4
model = MulticlassClassifier(input_dim, num_classes)
class_labels = [b'diboson', b'htautau', b'ttbar', b'ztautau']
class_labels = [label.decode('utf-8') for label in class_labels]  # Convert bytes to strings

# Training Loop

training = "test"

epochs = 10
save_path = os.path.join( user.model_directory, "multiClass", training) 
max_batch = 1

output_path = os.path.join(user.plot_directory, "multiClass", training)
os.makedirs(output_path, exist_ok=True)


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train_one_epoch(data_loader, class_labels, max_batch=max_batch)
    model.save(save_path, epoch)  # Save model after each epoch

    # Accumulate histograms
    true_histograms, pred_histograms, bin_edges = model.accumulate_histograms(
        data_loader, class_labels, max_batch=max_batch
    )

    # Plot convergence
    model.plot_convergence(
        true_histograms,
        pred_histograms,
        bin_edges,
        epoch,
        output_path,
        class_labels,
        feature_names,  # Pass feature names
    )

    # Evaluate on the same data for simplicity (use a validation set in practice)
    model.evaluate(data_loader, class_labels, max_batch=max_batch)

# Load the saved model for further use
model.load(save_path, checkpoint=5)
model.load(save_path)
common.syncer.sync()

