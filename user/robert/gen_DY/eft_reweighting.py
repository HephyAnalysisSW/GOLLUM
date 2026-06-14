import os
import re

import awkward as ak
import numpy as np


def _parse_weight_indices(path, text=None):
    if text is None:
        text = open(path).read()
    matches = re.findall(
        r'"([^"]+)"\s*:\s*\{\s*\n\s*"func"\s*:\s*lambda events: events\.LHEReweightingWeight\[:,\s*(\d+)\]',
        text,
    )
    if not matches:
        raise RuntimeError(f"Could not parse EFT weight mapping from {path}")
    return {name: int(index) for name, index in matches}


def _load_short_weight_indices():
    path = os.path.join(os.path.dirname(__file__), "info", "eft_weights.txt")
    text = open(path).read()
    return _parse_weight_indices(path, text=text)


def _load_full_weight_indices():
    path = os.path.join(os.path.dirname(__file__), "info", "weights_config.py")
    text = open(path).read()
    starts = [match.start() for match in re.finditer(r"\n\s*variables\s*=", text)]
    if not starts:
        raise RuntimeError(f"Could not find variables blocks in {path}")
    end = starts[1] if len(starts) > 1 else len(text)
    return _parse_weight_indices(path, text=text[starts[0] : end])


def _wc_names(weight_index):
    names = []
    for name in weight_index:
        if name == "sm" or name.endswith("_m1") or "_" in name:
            continue
        names.append(name)
    return sorted(names)


EFT_WEIGHT_INDICES = {
    "short": _load_short_weight_indices(),
    "full": _load_full_weight_indices(),
}
EFT_WC_NAMES_BY_CONFIG = {key: _wc_names(value) for key, value in EFT_WEIGHT_INDICES.items()}

EFT_WEIGHT_INDEX = EFT_WEIGHT_INDICES["short"]
EFT_WC_NAMES = EFT_WC_NAMES_BY_CONFIG["short"]
N_EFT_WEIGHTS = max(EFT_WEIGHT_INDEX.values()) + 1


def _required_labels(active):
    labels = ["sm"]
    names = list(active)
    for op in names:
        labels.extend([op, f"{op}_m1"])
    for i, op1 in enumerate(names):
        for op2 in names[i + 1 :]:
            labels.append(f"{op1}_{op2}")
            labels.append(f"{op2}_{op1}")
    return labels


def _select_weight_index(wilson_coefficients, config="short", vector_length=None):
    active = {name: float(value) for name, value in wilson_coefficients.items() if float(value) != 0.0}
    configs = list(EFT_WEIGHT_INDICES) if config == "auto" else [config]
    candidates = []
    for key in configs:
        if key not in EFT_WEIGHT_INDICES:
            raise ValueError(f"Unknown EFT weight config '{key}'. Known: {', '.join(sorted(EFT_WEIGHT_INDICES))}, auto")
        weight_index = EFT_WEIGHT_INDICES[key]
        unknown = sorted(set(active) - set(EFT_WC_NAMES_BY_CONFIG[key]))
        if unknown:
            continue
        try:
            used_indices = _used_indices(active, weight_index)
        except ValueError:
            continue
        n_weights = max(used_indices) + 1
        full_vector_length = max(weight_index.values()) + 1
        if config == "auto" and vector_length is not None and full_vector_length != vector_length:
            continue
        if vector_length is None or n_weights <= vector_length:
            candidates.append((n_weights, key, weight_index))
    if not candidates:
        detail = f" for vector length {vector_length}" if vector_length is not None else ""
        raise ValueError(f"No EFT weight config can evaluate {sorted(active)}{detail}")
    candidates.sort()
    return candidates[-1][1], candidates[-1][2]


def _used_indices(active, weight_index):
    indices = [weight_index["sm"]]
    for op in active:
        indices.extend([weight_index[op], weight_index[f"{op}_m1"]])
    names = list(active)
    for i, op1 in enumerate(names):
        for op2 in names[i + 1 :]:
            pair = f"{op1}_{op2}"
            if pair not in weight_index:
                pair = f"{op2}_{op1}"
            if pair not in weight_index:
                raise ValueError(f"No pairwise EFT weight for {op1}, {op2}")
            indices.extend([weight_index[pair], weight_index[op1], weight_index[op2], weight_index["sm"]])
    return indices


def eft_alpha_vector(config="short", **wilson_coefficients):
    """Return alpha such that w(c) = sum_k alpha_k(c) * LHEReweightingWeight[k].

    This is the quadratic interpolation with w(0)=LHEReweightingWeight[sm].
    The alpha representation is algebraically identical to constructing the
    linear, diagonal-quadratic, and mixed-quadratic coefficients event by event,
    but evaluates the event weight with a single dot product.
    """
    _, weight_index = _select_weight_index(wilson_coefficients, config=config)
    wc_names = _wc_names(weight_index)
    unknown = sorted(set(wilson_coefficients) - set(wc_names))
    if unknown:
        raise ValueError(f"Unknown Wilson coefficient(s): {', '.join(unknown)}")

    active = {name: float(value) for name, value in wilson_coefficients.items() if float(value) != 0.0}
    n_weights = max(_used_indices(active, weight_index)) + 1
    alpha = np.zeros(n_weights, dtype=np.float64)
    alpha[weight_index["sm"]] = 1.0

    for op, c in active.items():
        plus = weight_index[op]
        minus = weight_index[f"{op}_m1"]
        alpha[plus] += 0.5 * c + 0.5 * c * c
        alpha[minus] += -0.5 * c + 0.5 * c * c
        alpha[weight_index["sm"]] += -c * c

    active_names = list(active)
    for i, op1 in enumerate(active_names):
        for op2 in active_names[i + 1 :]:
            pair = f"{op1}_{op2}"
            if pair not in weight_index:
                pair = f"{op2}_{op1}"
            if pair not in weight_index:
                raise ValueError(f"No pairwise EFT weight for {op1}, {op2}")
            c12 = active[op1] * active[op2]
            alpha[weight_index[pair]] += c12
            alpha[weight_index[op1]] -= c12
            alpha[weight_index[op2]] -= c12
            alpha[weight_index["sm"]] += c12

    return alpha


def _weight_matrix(events_or_weights, n_weights):
    if hasattr(events_or_weights, "fields") and "LHEReweightingWeight" in events_or_weights.fields:
        weights = events_or_weights["LHEReweightingWeight"]
    else:
        weights = events_or_weights
    lengths = ak.to_numpy(ak.num(weights, axis=1))
    if np.any(lengths < n_weights):
        raise ValueError(
            f"Expected all LHEReweightingWeight vectors to have at least {n_weights} entries, "
            f"got min length {int(np.min(lengths))}"
        )
    matrix = ak.to_numpy(ak.to_regular(weights[:, :n_weights]))
    if matrix.ndim != 2 or matrix.shape[1] < n_weights:
        raise ValueError(f"Expected LHEReweightingWeight with at least {n_weights} entries, got shape {matrix.shape}")
    return np.asarray(matrix[:, :n_weights], dtype=np.float64)


def eft_weight_function(config="short", **wilson_coefficients):
    """Return a function evaluating the absolute LHE EFT weight w(c)."""
    alpha = eft_alpha_vector(config=config, **wilson_coefficients)

    def weight(events_or_weights):
        return _weight_matrix(events_or_weights, len(alpha)).dot(alpha)

    return weight


def eft_weight(events_or_weights, config="short", **wilson_coefficients):
    """Return the LHE EFT reweight factor w(c).

    For these SMEFT NanoAOD samples LHEReweightingWeight already stores the
    event reweight factor. It must not be divided by LHEWeight_originalXWGTUP.
    """
    if config == "auto":
        lengths = ak.to_numpy(ak.num(events_or_weights, axis=1))
        out = np.full(len(lengths), np.nan, dtype=np.float64)
        for length in sorted(set(map(int, lengths))):
            mask = lengths == length
            key, _ = _select_weight_index(wilson_coefficients, config="auto", vector_length=length)
            out[mask] = eft_weight(events_or_weights[mask], config=key, **wilson_coefficients)
        return out
    alpha = eft_alpha_vector(config=config, **wilson_coefficients)
    return _weight_matrix(events_or_weights, len(alpha)).dot(alpha)


def parse_eft_point(text):
    if ":" in text:
        label, body = text.split(":", 1)
        label = label.strip()
    else:
        label, body = "", text

    values = {}
    body = body.strip()
    if body:
        for item in body.split(","):
            key, value = item.split("=", 1)
            values[key.strip()] = float(value)

    if not label:
        label = "SM" if not values else ",".join(f"{key}={value:g}" for key, value in values.items())
    return label, values
