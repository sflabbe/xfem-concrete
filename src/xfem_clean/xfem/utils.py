"""Small utilities shared by analysis drivers."""

from __future__ import annotations

import numpy as np


def find_nearest_node(nodes: np.ndarray, x: float, y: float) -> int:
    """Return index of the node closest to (x, y) in Euclidean distance."""
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError(f"nodes must be (n,2+) array, got shape={nodes.shape}")
    dx = nodes[:, 0] - float(x)
    dy = nodes[:, 1] - float(y)
    return int(np.argmin(dx * dx + dy * dy))
