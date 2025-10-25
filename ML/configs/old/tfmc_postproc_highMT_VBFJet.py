''' A config for inclusive cross section parametrization, quadratic in pdf. 
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

import sys
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')
sys.path.insert( 0, '.')

# Always the same:
from configs.common import *
import common.data_structure as data_structure

n_epochs         = 300
n_epochs_phaseout= 100  # This number of epochs  at the end for phaseout 

# Initial learning rate
learning_rate = 0.002

had_pt  = data_structure.feature_names.index("PRI_had_pt")
had_eta = data_structure.feature_names.index("PRI_had_eta")
had_phi = data_structure.feature_names.index("PRI_had_phi")
lep_pt  = data_structure.feature_names.index("PRI_lep_pt")
lep_eta = data_structure.feature_names.index("PRI_lep_eta")
lep_phi = data_structure.feature_names.index("PRI_lep_phi")
met     = data_structure.feature_names.index("PRI_met")
met_phi = data_structure.feature_names.index("PRI_met_phi")
mass_transverse_met_lep = data_structure.feature_names.index("DER_mass_transverse_met_lep")
mass_vis                = data_structure.feature_names.index("DER_mass_vis")

def build_model(input_dim=10, hidden_layers=[64,128,64], activation='relu', num_classes=2):
    """Build a simple neural network for classification with L1/L2 regularization and dropout."""

    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    # Hidden layers with L1/L2 regularization and dropout
    for units in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=None,
            )
        )
        model.add(tf.keras.layers.Activation(activation))  # Apply activation

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
        )
    )
    model.trainable = False  # Ensure weights remain frozen

    return model

preprocessor_network = build_model()

# Load the latest checkpoint

checkpoint_dir = "/groups/hephy/cms/robert.schoefbeck/Challenge/models/TFMC/highMT_VBFJet/tfmc_2_reg_preproc/v6"
checkpoint = tf.train.Checkpoint(model=preprocessor_network)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Restoring weights from {latest_checkpoint}")
    checkpoint.restore(latest_checkpoint).expect_partial()
else:
    raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_dir}")

def preprocessor( features, features_norm ):

    network_input = np.column_stack((
        features_norm[:, had_pt],
        features_norm[:, lep_pt],
        features_norm[:, met],
        features_norm[:, met_phi],
        features[:, had_eta] - features[:, lep_eta],
        features[:, had_phi] - features[:, lep_phi],
        features[:, met_phi] - features[:, lep_phi],
        features[:, met_phi] - features[:, had_phi],
        features_norm[:, mass_transverse_met_lep],
        features_norm[:, mass_vis],
    ))

    # Ensure the input is a TensorFlow tensor
    network_input_tensor = tf.convert_to_tensor(network_input, dtype=tf.float32)

    # Pass through the frozen network
    network_output = tf.stop_gradient(preprocessor_network(network_input_tensor))

    # Convert the network output back to NumPy if necessary
    network_output_np = network_output.numpy()


    combined_features = np.column_stack((
        network_output_np,    # Output from the frozen network
        features_norm,
    ))

    return combined_features

n_epochs           = 300
n_epochs_phaseout  = 100
learning_rate = 0.005

classes       = data_structure.labels
input_dim     = len(data_structure.feature_names) + preprocessor_network.output_shape[1] 
hidden_layers = [64,64,64]
activation    = 'relu'

#l1_reg        = 0.1
#l2_reg        = 0.05
#dropout_rate  = 0.2

use_ic        = True
use_scaler    = True
