import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from Multiclassifier import MulticlassClassifier

from data import get_data_loader
import common.features as features
import common.user as user
import common.syncer

data_loader = get_data_loader(
    n_split = 20, 
    selection_function = None,
    data_directory = "/eos/vbc/group/cms/robert.schoefbeck/Higgs_uncertainty/data/VBF/split_train_dataset/" )

# Initialize model
class_labels = features.class_labels 
model = MulticlassClassifier(len(features.feature_names), len(class_labels))

# Training Loop

training = "VBF"

epochs = 100
save_path = os.path.join( user.model_directory, "multiClass", training) 
max_batch = -1

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
        features.feature_names,  # Pass feature names
    )

    # Evaluate on the same data for simplicity (use a validation set in practice)
    model.evaluate(data_loader, class_labels, max_batch=max_batch)

# Load the saved model for further use
model.load(save_path)
common.syncer.sync()

