from __future__ import annotations

import numpy as np


def make_synthetic(client_id: str, num_samples: int, num_features: int):
    seed = abs(hash(client_id)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    w_true = rng.normal(size=(num_features,))
    x = rng.normal(size=(num_samples, num_features))
    noise = rng.normal(scale=0.1, size=(num_samples,))
    y = x @ w_true + noise
    return x, y
