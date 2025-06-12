import numpy as np

def quantize(W, b, q):
    """
    Quantize weight matrix W and bias b to the precision specified in self.config['quantization']:
      • None → full‑precision
      • 2    → binary (±1)
      • 3    → ternary (0, ±1)
      • n>=4 → uniform signed fixed‑point with n bits
    Returns:
        W_q, b_q: quantized versions of W and b
    """
    # no quantization
    if q is None:
        return W, b

    # binary: ±1
    if q == 2:
        W_q = np.sign(W)
        W_q[W_q == 0] = 1
        b_q = np.sign(b) if b != 0 else 1
        return W_q, b_q

    # ternary: -1, 0, +1 with threshold at mean magnitude
    if q == 3:
        thresh = np.mean(np.abs(W))
        W_q = np.where(np.abs(W) < thresh, 0, np.sign(W))
        b_q = 0 if abs(b) < thresh else np.sign(b)
        return W_q, b_q

    # n-bit uniform signed fixed-point (n>=4)
    # map W into integer range [-Q, +Q] then back
    Q = 2**(q-1) - 1
    max_w = np.max(np.abs(W))
    # avoid div by zero
    scale = Q / max_w if max_w != 0 else 1.0

    W_int = np.round(W * scale)
    W_int = np.clip(W_int, -Q, Q)
    W_q = W_int / scale

    # bias quantized with same scale
    b_int = np.round(b * scale)
    b_int = np.clip(b_int, -Q, Q)
    b_q = b_int / scale

    return W_q, b_q

