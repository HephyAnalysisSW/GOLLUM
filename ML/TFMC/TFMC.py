#!/usr/bin/env python
# Minimal TF multiclass classifier wrapper: no I/O, no plotting, no loop.

from __future__ import annotations
import os
import pickle
import numpy as np
import tensorflow as tf

class PhaseoutScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr: float, n_epochs: int, n_epochs_phaseout: int):
        self.initial_lr = float(initial_lr)
        self.n_epochs = int(n_epochs)
        self.n_epochs_phaseout = int(n_epochs_phaseout)

    def __call__(self, epoch):
        epoch = tf.cast(epoch, tf.float32)
        if self.n_epochs_phaseout <= 0:
            return tf.convert_to_tensor(self.initial_lr, dtype=tf.float32)
        cutoff = self.n_epochs - self.n_epochs_phaseout
        return tf.where(
            epoch < cutoff,
            tf.convert_to_tensor(self.initial_lr, tf.float32),
            tf.convert_to_tensor(self.initial_lr, tf.float32)
            - (tf.cast(epoch - cutoff, tf.float32) * (self.initial_lr / self.n_epochs_phaseout)),
        )

class TFMC:
    def __init__(
        self,
        input_dim: int,
        classes: list[str],
        activation: str = "relu",
        hidden_layers: list[int] = (64, 64, 64),
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-3,
        n_epochs: int = 1,
        n_epochs_phaseout: int = 0,
        reweighting: bool = True,
    ):
        self.input_dim = int(input_dim)
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.activation = activation
        self.hidden_layers = list(hidden_layers)
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.n_epochs_phaseout = int(n_epochs_phaseout)
        self.reweighting = bool(reweighting)

        self.feature_means = np.zeros(self.input_dim, dtype=np.float64)
        self.feature_variances = np.ones(self.input_dim, dtype=np.float64)
        # IC class weights: default flat if not provided
        self.class_weights = np.ones(self.num_classes, dtype=np.float64)

        self.model = self._build_model()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.lr_schedule = PhaseoutScheduler(self.learning_rate, self.n_epochs, self.n_epochs_phaseout)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    # ---------------- model ----------------
    def _build_model(self) -> tf.keras.Model:
        from tensorflow.keras import regularizers, layers, Sequential, initializers
        reg = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg) if (self.l1_reg > 0 or self.l2_reg > 0) else None
        m = Sequential()
        m.add(layers.Input(shape=(self.input_dim,)))
        for units in self.hidden_layers:
            m.add(layers.Dense(units, activation=self.activation, kernel_regularizer=reg))
            if self.dropout_rate and self.dropout_rate > 0:
                m.add(layers.Dropout(self.dropout_rate))
        # Start from equal class logits so the initial softmax is flat.
        # With the existing DCR conversion (divide by class_weights, then renormalize),
        # this makes the epoch-0 inclusive fractions reflect the class-weight sums.
        m.add(
            layers.Dense(
                self.num_classes,
                activation="softmax",
                kernel_regularizer=reg,
                kernel_initializer=initializers.Zeros(),
                bias_initializer=initializers.Zeros(),
            )
        )
        return m

    # ---------------- scalers / IC ----------------
    def set_scaler(self, means: np.ndarray, variances: np.ndarray):
        self.feature_means = np.asarray(means, dtype=np.float64)
        self.feature_variances = np.asarray(variances, dtype=np.float64)

    def set_ic_weights_from_sums(self, class_order: list[str], weight_sums: dict[int | str, float]):
        """
        class_order: list of class names (same as self.classes). weight_sums per class label/order.
        Will compute scaling factors mean / class_sum = total / (n_classes * class_sum)
        """
        # Accept dict keyed by class name or by integer index (0..C-1)
        vals = []
        for i, name in enumerate(class_order):
            if name in weight_sums:
                vals.append(float(weight_sums[name]))
            elif i in weight_sums:
                vals.append(float(weight_sums[i]))
            else:
                raise RuntimeError(f"Missing IC weight for class '{name}'.")
        vals = np.asarray(vals, dtype=np.float64)
#        total = np.sum(vals)
#        self.class_weights = np.where(vals > 0, total / vals, 1.0)
        mean = np.mean(vals)
        self.class_weights = np.where(vals > 0, mean / vals, 1.0)
    # ---------------- inference ----------------
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_means) / np.sqrt(self.feature_variances)

    # In contrast to GOLLUM, we report either the probability ratio or the DCR. We do not report 1/classweights*probability which has yet to be normalized.
    def predict(self, X: np.ndarray, probability: bool = False) -> np.ndarray:
        Xn = self._normalize(X)
        y = self.model(Xn, training=False).numpy()
        if probability:
            return y
        else: #DCR
            y /= self.class_weights
            return y / (y.sum(axis=1, keepdims=True) + 1e-12)

    # ---------------- training primitives ----------------
    @tf.function
    def _train_step_tf(self, X, y_onehot, w):
        with tf.GradientTape() as tape:
            pred = self.model(X, training=True)
            #print("hello:lower",pred[X[:,0]<0.5], )
            #print("hello:higher",pred[X[:,0]>0.5], )
            loss_per = self.loss_fn(y_onehot, pred)  # shape [N]
            if w is not None:
                loss = tf.reduce_sum(loss_per * w) / (tf.reduce_sum(w) + 1e-12)
            else:
                loss = tf.reduce_mean(loss_per)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train_on_batch(self, X: np.ndarray, y_onehot: np.ndarray, w: np.ndarray | None):
        # normalize and apply per-class reweighting if desired
        Xn = self._normalize(X).astype(np.float32, copy=False)
        w_eff = None
        if w is not None:
            w = w.astype(np.float32, copy=False)
            if self.reweighting:
                # multiply by class factors based on argmax of onehot
                cls = np.argmax(y_onehot, axis=1)
                w = w * self.class_weights[cls].astype(np.float32)
            w_eff = w
        loss = self._train_step_tf(tf.convert_to_tensor(Xn), tf.convert_to_tensor(y_onehot, dtype=tf.float32),
                                   None if w_eff is None else tf.convert_to_tensor(w_eff))
        return float(loss.numpy())

    # --------------- validation primitives (no weight update) ----------------- #
    @tf.function
    def _loss_step_tf(self, X, y_onehot, w):
        """Compute loss only (no gradients, no updates)."""
        pred = self.model(X, training=False)  # training=False disables dropout/batch norm
        loss_per = self.loss_fn(y_onehot, pred)
        if w is not None:
            loss = tf.reduce_sum(loss_per * w) / (tf.reduce_sum(w) + 1e-12)
        else:
            loss = tf.reduce_mean(loss_per)
        return loss

    def compute_loss(self, X: np.ndarray, y_onehot: np.ndarray, w: np.ndarray | None) -> float:
        """Public method for validation/test loss (no weight updates)."""
        Xn = self._normalize(X).astype(np.float32, copy=False)
        w_eff = w.astype(np.float32, copy=False) if w is not None else None
        if w is not None:
            w = w.astype(np.float32, copy=False)
            if self.reweighting:
                # multiply by class factors based on argmax of onehot
                cls = np.argmax(y_onehot, axis=1)
                w = w * self.class_weights[cls].astype(np.float32)
            w_eff = w
        loss = self._loss_step_tf(tf.convert_to_tensor(Xn), tf.convert_to_tensor(y_onehot, dtype=tf.float32),
                                None if w_eff is None else tf.convert_to_tensor(w_eff))
        return float(loss.numpy())        

    # ---------------- checkpointing ----------------
    def save(self, save_dir: str, epoch: int | None = None, is_best: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        if epoch is None:
            epoch = 0
        ckpt_path = os.path.join(save_dir, str(int(epoch)))
        self.checkpoint.write(ckpt_path)
        meta = dict(
            classes=self.classes,
            input_dim=self.input_dim,
            feature_means=self.feature_means,
            feature_variances=self.feature_variances,
            class_weights=self.class_weights,
            activation=self.activation,
            hidden_layers=self.hidden_layers,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            n_epochs_phaseout=self.n_epochs_phaseout,
        )
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(meta, f)
        
        # allow loading last epoch or best epoch
        # not the same if Early Stopping is engaged and patience > 0
        with open(os.path.join(save_dir, "last_checkpoint"), "w") as f:
            f.write(f'model_checkpoint_path: "{ckpt_path}"\n')
        if is_best:
            with open(os.path.join(save_dir, "checkpoint"), "w") as f:
                f.write(f'model_checkpoint_path: "{ckpt_path}"\n')            
        #print("Written to", os.path.join(save_dir, "checkpoint"))

    @classmethod
    # default will load the model at its best epoch (see above)
    def load(cls, save_dir: str, latest_filename: str="checkpoint") -> "TFMC":
        latest = tf.train.latest_checkpoint(save_dir, latest_filename=latest_filename)
        if not latest:
            raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        with open(os.path.join(save_dir, "config.pkl"), "rb") as f:
            meta = pickle.load(f)
        inst = cls(
            input_dim=meta["input_dim"],
            classes=meta["classes"],
            activation=meta["activation"],
            hidden_layers=meta["hidden_layers"],
            l1_reg=meta["l1_reg"],
            l2_reg=meta["l2_reg"],
            dropout_rate=meta["dropout_rate"],
            learning_rate=meta["learning_rate"],
            n_epochs=meta["n_epochs"],
            n_epochs_phaseout=meta["n_epochs_phaseout"],
        )
        inst.feature_means = np.asarray(meta["feature_means"])
        inst.feature_variances = np.asarray(meta["feature_variances"])
        inst.class_weights = np.asarray(meta["class_weights"])
        inst.checkpoint.restore(latest).expect_partial()
        return inst
