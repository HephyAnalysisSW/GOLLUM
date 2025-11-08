import os
import pickle
import importlib
import numpy as np
import operator, functools
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

class PNN:
    """
    Parametric Neural Net (local DCR) — model & IO only.
    Training loop, data iteration, and plotting live in pnn_training.py
    """

    def __init__(self,
                 parameters,
                 combinations,
                 base_points,
                 input_dim,
                 hidden_layers=(128,128),
                 activation="relu",
                 learning_rate=1e-3,
                 n_epochs=200,
                 n_epochs_phaseout=0,
                 initialize_zero=False):
        # config-ish
        self.parameters         = list(parameters)
        self.combinations       = [tuple(c) for c in combinations]
        self.base_points        = np.asarray(base_points, dtype=float)
        self.input_dim          = int(input_dim)
        self.hidden_layers      = list(hidden_layers)
        self.activation         = str(activation)
        self.learning_rate      = float(learning_rate)
        self.n_epochs           = int(n_epochs)
        self.n_epochs_phaseout  = int(n_epochs_phaseout)
        self.initialize_zero    = bool(initialize_zero)

        # find nominal index (all zeros)
        z = np.zeros(len(self.parameters), dtype=float)
        idx = np.where(np.all(np.isclose(self.base_points, z), axis=1))[0]
        if len(idx) == 0:
            raise RuntimeError("PNN: no nominal base point (all zeros) found in base_points.")
        self.nominal_base_point_index = int(idx[0])

        # build VkA and C inverse (same math as before)
        self.VkA = np.zeros((len(self.base_points), len(self.combinations)), dtype=np.float32)
        for i, nu in enumerate(self.base_points):
            for j, comb in enumerate(self.combinations):
                self.VkA[i, j] = functools.reduce(
                    operator.mul, (nu[self.parameters.index(c)] for c in comb), 1.0
                )

        mask = np.ones(len(self.base_points), bool)
        mask[self.nominal_base_point_index] = False
        self.masked_base_points = self.base_points[mask]

        C = np.zeros((len(self.combinations), len(self.combinations)), dtype=np.float64)
        for nu in self.masked_base_points:
            v = np.array([functools.reduce(operator.mul, (nu[self.parameters.index(c)] for c in comb), 1.0)
                          for comb in self.combinations], dtype=np.float64)
            C += np.outer(v, v)
        if np.linalg.matrix_rank(C) != C.shape[0]:
            raise RuntimeError("PNN: base-point matrix C is rank-deficient. Check base_points/combinations.")
        self.CInv = np.linalg.inv(C)

        # model
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # scaler placeholders (optionally set via set_scaler)
        self.feature_means     = None
        self.feature_variances = None

        # ICP bias (ΔA-space), applied additively to model outputs
        self._icp_bias = None  # tf.Tensor shape (C,)

        # checkpoint wrapper
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    # ---------------------- utils ----------------------
    def _build_model(self):
        l1 = 0.0
        l2 = 0.0
        reg = regularizers.l1_l2(l1=l1, l2=l2) if (l1 > 0 or l2 > 0) else None

        m = tf.keras.Sequential()
        m.add(tf.keras.layers.Input(shape=(self.input_dim,)))
        for units in self.hidden_layers:
            m.add(Dense(units, activation=self.activation, kernel_regularizer=reg))
        if self.initialize_zero:
            m.add(Dense(len(self.combinations),
                        activation=None,
                        kernel_initializer=tf.keras.initializers.Zeros(),
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_regularizer=reg))
        else:
            m.add(Dense(len(self.combinations), activation=None, kernel_regularizer=reg))
        return m

    def set_scaler(self, means: np.ndarray, variances: np.ndarray):
        self.feature_means = np.asarray(means, dtype=np.float64)
        self.feature_variances = np.asarray(variances, dtype=np.float64)
        if self.feature_means.shape[0] != self.input_dim or self.feature_variances.shape[0] != self.input_dim:
            raise ValueError("PNN.set_scaler: shape mismatch with input_dim.")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_means is None or self.feature_variances is None:
            return X
        return (X - self.feature_means) / np.sqrt(self.feature_variances)

    def nu_A(self, nu_vec):
        nu = list(nu_vec)
        return np.array([
            functools.reduce(operator.mul, (nu[self.parameters.index(c)] for c in comb), 1.0)
            for comb in self.combinations
        ], dtype=np.float64)

    # ---------------------- ICP bias ----------------------
    def set_icp(self, parameters, combinations, DeltaA):
        """
        Store ICP bias in ΔA-space so the effective output is: ΔA_net(x) + ΔA_icp.
        Requires identical parameters & (non-empty) combinations order as this PNN.
        """
        import numpy as _np
        # parameters must match exactly
        if list(parameters) != list(self.parameters):
            raise ValueError(f"ICP parameters mismatch: {parameters} vs {self.parameters}")
        # compare non-empty combinations only (model outputs those)
        pnn_combs = [tuple(c) for c in self.combinations if len(c) > 0]
        icp_combs = [tuple(c) for c in combinations     if len(c) > 0]
        if icp_combs != pnn_combs:
            raise ValueError(f"ICP combinations mismatch:\nICP: {icp_combs}\nPNN: {pnn_combs}")
        DeltaA = _np.asarray(DeltaA, dtype=_np.float32).reshape(-1)
        if DeltaA.shape[0] != len(pnn_combs):
            raise ValueError(f"DeltaA length {DeltaA.shape[0]} != #combos {len(pnn_combs)}")
        self._icp_bias = tf.constant(DeltaA, dtype=tf.float32)

    # ---------------------- inference helpers ----------------------
    def deltaA(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        out = self.model(tf.convert_to_tensor(Xn, dtype=tf.float32), training=False)
        if self._icp_bias is not None:
            out = out + self._icp_bias
        return out.numpy()

    def deltaA_tf(self, X_tf: tf.Tensor, training: bool) -> tf.Tensor:
        """X_tf must be already normalized."""
        out = self.model(X_tf, training=training)
        if self._icp_bias is not None:
            out = out + self._icp_bias
        return out

    def predict_ratio(self, X: np.ndarray, nu_vec) -> np.ndarray:
        """Return local DCR at features X and nuisance values nu."""
        dA = self.deltaA(X)
        return np.exp(dA @ vk)

    # ---------------------- IO ----------------------
    def save(self, save_dir, epoch: int):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, str(epoch))
        self.checkpoint.write(ckpt_path)
        with open(os.path.join(save_dir, 'checkpoint'), 'w') as f:
            f.write(f'model_checkpoint_path: "{ckpt_path}"\n')

        payload = dict(
            parameters=self.parameters,
            combinations=self.combinations,
            base_points=self.base_points,
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            n_epochs_phaseout=self.n_epochs_phaseout,
            initialize_zero=self.initialize_zero,
            feature_means=self.feature_means,
            feature_variances=self.feature_variances,
            icp_bias=None if self._icp_bias is None else self._icp_bias.numpy(),
        )
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, save_dir):
        cfg_path = os.path.join(save_dir, "config.pkl")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.pkl in {save_dir}")
        with open(cfg_path, "rb") as f:
            payload = pickle.load(f)

        p = cls(**{k: payload[k] for k in [
            "parameters","combinations","base_points","input_dim",
            "hidden_layers","activation","learning_rate","n_epochs",
            "n_epochs_phaseout","initialize_zero"
        ]})
        if payload.get("feature_means") is not None:
            p.set_scaler(payload["feature_means"], payload["feature_variances"])

        latest = tf.train.latest_checkpoint(save_dir)
        if not latest:
            raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        p.checkpoint.restore(latest).expect_partial()

        icp_bias = payload.get("icp_bias", None)
        if icp_bias is not None:
            p._icp_bias = tf.constant(icp_bias, dtype=tf.float32)

        return p

