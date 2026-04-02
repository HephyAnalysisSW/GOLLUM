import os
import pickle
import numpy as np
import importlib
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense


class DNN:
    """
    DNN binary classifier — model & IO only.
    Training loop and data iteration live in dnn_training.py
    """

    def __init__(self,
                 input_dim,
                 hidden_layers=(512, 512, 256, 128),
                 activation="relu",
                 learning_rate=1e-3,
                 l1=0.0,   # L1 regulator
                 l2=0.0,   # L2 regulator
                 output_activation=None,  # None -> logits; "sigmoid" -> prob
                 ):
        # config-ish
        self.input_dim         = int(input_dim)
        self.hidden_layers     = list(hidden_layers)
        self.activation        = str(activation)
        self.learning_rate     = float(learning_rate)
        self.l1                = float(l1)
        self.l2                = float(l2)
        self.output_activation = output_activation

        # model
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # scaler placeholders (optionally set via set_scaler)
        self.feature_means     = None
        self.feature_variances = None

        # checkpoint wrapper (match PNN style)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    # ---------------------- utils ----------------------
    def _build_model(self):

        if (self.l1 > 0 or self.l2 > 0):
            reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)
            print(f"Build DNN with regulators L1={self.l1} and L2={self.l2}")
        else:
            reg = None
        m = tf.keras.Sequential()
        m.add(tf.keras.layers.Input(shape=(self.input_dim,), dtype=tf.float32))
        for units in self.hidden_layers:
            m.add(Dense(units, activation=self.activation, kernel_regularizer=reg))
        m.add(Dense(1, activation=self.output_activation, kernel_regularizer=reg))
        return m

    def set_scaler(self, means: np.ndarray, variances: np.ndarray):
        self.feature_means = np.asarray(means, dtype=np.float32)
        self.feature_variances = np.asarray(variances, dtype=np.float32)
        if self.feature_means.shape[0] != self.input_dim or self.feature_variances.shape[0] != self.input_dim:
            raise ValueError("DNN.set_scaler: shape mismatch with input_dim.")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_means is None or self.feature_variances is None:
            return X
        return (X - self.feature_means) / np.sqrt(self.feature_variances)

    # ---------------------- inference helpers ----------------------
    def logits(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X).astype(np.float32, copy=False)
        out = self.model(Xn, training=False)   # 直接喂 numpy，TF 会零拷贝/少拷贝处理
        return out.numpy()

    def logits_tf(self, X_tf: tf.Tensor, training: bool) -> tf.Tensor:
        """
        TensorFlow forward. Convention matches PNN.deltaA_tf:
        X_tf must already be normalized if you use a scaler.
        """
        return self.model(X_tf, training=training)

    # ---------------------- IO ----------------------
    def save(self, save_dir: str, epoch: int):
        """
        Save checkpoint + config.pkl (PNN-compatible pattern).
        """
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, str(epoch))
        self.checkpoint.write(ckpt_path)

        # create TF-style 'checkpoint' file to help tf.train.latest_checkpoint
        with open(os.path.join(save_dir, "checkpoint"), "w") as f:
            f.write(f'model_checkpoint_path: "{ckpt_path}"\n')

        payload = dict(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            learning_rate=self.learning_rate,
            l1=self.l1,
            l2=self.l2,
            output_activation=self.output_activation,
            feature_means=self.feature_means,
            feature_variances=self.feature_variances,
        )
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, save_dir: str):
        """
        Load from config.pkl + latest checkpoint.
        """
        cfg_path = os.path.join(save_dir, "config.pkl")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.pkl in {save_dir}")
        with open(cfg_path, "rb") as f:
            payload = pickle.load(f)

        d = cls(**{k: payload[k] for k in [
            "input_dim", "hidden_layers", "activation", "learning_rate",
            "l1", "l2", "output_activation"
        ]})

        if payload.get("feature_means") is not None:
            d.set_scaler(payload["feature_means"], payload["feature_variances"])

        latest = tf.train.latest_checkpoint(save_dir)
        if not latest:
            raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        d.checkpoint.restore(latest).expect_partial()

        return d
