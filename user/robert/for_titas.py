import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate toy data: two one-dimensional exponential distributions
np.random.seed(42)
N_0, N_1 = 10000, 10000
lambda_0, lambda_1 = 1.0, 0.5

z_0 = np.random.exponential(scale=1 / lambda_0, size=N_0).reshape(-1, 1).astype(np.float32)
w_0 = np.ones(N_0, dtype=np.float32)  # Equal weights for simplicity

z_1 = np.random.exponential(scale=1 / lambda_1, size=N_1).reshape(-1, 1).astype(np.float32)
w_1 = np.ones(N_1,dtype=np.float32)

# Combine the datasets_hephy and labels
z = np.concatenate([z_0, z_1], axis=0)
w = np.concatenate([w_0, w_1], axis=0)
y = np.concatenate([np.zeros(N_0), np.ones(N_1)]).astype(np.float32)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((z, y, w)).shuffle(len(z)).batch(128)

# Define the neural network model
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize the model
input_dim = z.shape[1]
model = create_model(input_dim)

# Define custom loss function
def custom_loss(y_true, y_pred, weights):
    term_1 = tf.reduce_sum(weights * y_true * -tf.math.log(y_pred + 1e-8))
    term_0 = tf.reduce_sum(weights * (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
    return term_1 + term_0

# Training step
@tf.function
def train_step(x, y, weights):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = custom_loss(y, predictions, weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch, w_batch in dataset:
        loss = train_step(x_batch, tf.expand_dims(y_batch, axis=-1), w_batch)
        epoch_loss += loss.numpy()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# Compute r(z)
def compute_r(z):
    f_z = model.predict(z)
    return f_z / (1 - f_z + 1e-8)

r_z_0 = compute_r(z_0)
r_z_1 = compute_r(z_1)

# Reweighting and closure test
weights_0_to_1 = w_0 * r_z_0.flatten()
weights_1_to_0 = w_1 / (r_z_1.flatten() + 1e-8)

# Make histograms
bins = np.linspace(0, np.max([z_0.max(), z_1.max()]), 50)
plt.figure(figsize=(12, 6))

# Original distributions
plt.hist(z_0, bins=bins, weights=w_0, alpha=0.5, label="Original $z_0$ (dσ₀)", density=True)
plt.hist(z_1, bins=bins, weights=w_1, alpha=0.5, label="Original $z_1$ (dσ₁)", density=True)

# Reweighted distributions
plt.hist(z_0, bins=bins, weights=weights_0_to_1, histtype='step', lw=2, label="Reweighted $z_0 \\to dσ₁$", density=True)
plt.hist(z_1, bins=bins, weights=weights_1_to_0, histtype='step', lw=2, label="Reweighted $z_1 \\to dσ₀$", density=True)

# Plot settings
plt.xlabel("z")
plt.ylabel("Density")
plt.legend()
plt.title("Closure Test: Reweighting Distributions")
plt.grid()
plt.show()

plt.savefig("closure_test.png")

