"""NumPy compatibility utilities for different numpy versions."""

import numpy as np


def trapezoid(y, x):
    """
    Trapezoidal integration compatible with old and new numpy.

    NumPy >= 2.0 has np.trapezoid, older versions use np.trapz.
    This wrapper provides a unified interface.

    Parameters
    ----------
    y : array_like
        Values to integrate
    x : array_like
        Sample points corresponding to y values

    Returns
    -------
    float
        Integral of y(x) using the trapezoidal rule
    """
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn(y, x)
    return np.trapz(y, x)
