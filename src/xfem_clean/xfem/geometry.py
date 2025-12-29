"""Crack geometry and near-tip branch functions.

This module is intentionally self-contained and purely geometric; it does not
depend on FEM assembly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class XFEMCrack:
    """Single straight crack segment.

    The segment starts at (x0, y0) and ends at the current tip (tip_x, tip_y).
    """

    x0: float
    y0: float
    tip_x: float
    tip_y: float
    stop_y: float
    angle_deg: float = 90.0
    active: bool = False

    def p0(self) -> np.ndarray:
        return np.array([float(self.x0), float(self.y0)], dtype=float)

    def pt(self) -> np.ndarray:
        return np.array([float(self.tip_x), float(self.tip_y)], dtype=float)

    def tvec(self) -> np.ndarray:
        v = self.pt() - self.p0()
        n = float(np.linalg.norm(v))
        if n <= 1e-14:
            return np.array([0.0, 1.0], dtype=float)
        return (v / n).astype(float)

    def nvec(self) -> np.ndarray:
        t = self.tvec()
        n = np.array([-t[1], t[0]], dtype=float)
        nn = float(np.linalg.norm(n)) + 1e-15
        return (n / nn).astype(float)

    def s_tip(self) -> float:
        t = self.tvec()
        return float(np.dot(t, self.pt() - self.p0()))

    def s(self, x: float, y: float) -> float:
        t = self.tvec()
        return float(np.dot(t, np.array([float(x), float(y)], dtype=float) - self.p0()))

    def phi(self, x: float, y: float) -> float:
        """Signed distance to the infinite crack line (positive on +n side)."""
        n = self.nvec()
        return float(np.dot(n, np.array([float(x), float(y)], dtype=float) - self.p0()))

    def H(self, x: float, y: float) -> float:
        """Heaviside sign (+1/-1) based on phi."""
        return 1.0 if self.phi(x, y) >= 0.0 else -1.0

    def behind_tip(self, x: float, y: float) -> float:
        """Gate enrichment/cohesive to the existing segment only (0 <= s <= s_tip)."""
        s = self.s(x, y)
        return 1.0 if (s >= -1e-12 and s <= self.s_tip() + 1e-12) else 0.0

    def cuts_element(self, xe: np.ndarray) -> bool:
        """Return True if the current crack segment intersects the element bbox and changes sign."""
        if not self.active:
            return False
        xmin, xmax = float(xe[:, 0].min()), float(xe[:, 0].max())
        ymin, ymax = float(xe[:, 1].min()), float(xe[:, 1].max())
        seg = clip_segment_to_bbox(self.p0(), self.pt(), xmin, xmax, ymin, ymax)
        if seg is None:
            return False
        phis = [self.phi(float(xe[a, 0]), float(xe[a, 1])) for a in range(xe.shape[0])]
        return (min(phis) < 0.0) and (max(phis) > 0.0)


def tip_polar(x: float, y: float, xt: float, yt: float, r_eps: float = 1e-12):
    dx = float(x) - float(xt)
    dy = float(y) - float(yt)
    r = math.sqrt(dx * dx + dy * dy)
    r = max(r, float(r_eps))
    th = math.atan2(dy, dx)
    return r, th, dx, dy


def branch_F_and_grad(x: float, y: float, xt: float, yt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moës tip branch functions and their gradients.

    Returns
    -------
    F : (4,) ndarray
    dFdx : (4,) ndarray
    dFdy : (4,) ndarray
    """

    r, th, dx, dy = tip_polar(x, y, xt, yt)
    s = math.sqrt(r)

    sh = math.sin(th / 2.0)
    ch = math.cos(th / 2.0)
    sth = math.sin(th)
    cth = math.cos(th)

    F = np.array([s * sh, s * ch, s * sh * sth, s * ch * sth], dtype=float)

    drdx = dx / r
    drdy = dy / r
    dthdx = -dy / (r * r)
    dthdy = dx / (r * r)

    dFdr = np.zeros(4, dtype=float)
    dFdth = np.zeros(4, dtype=float)

    dFdr[0] = 0.5 / s * sh
    dFdth[0] = s * 0.5 * ch

    dFdr[1] = 0.5 / s * ch
    dFdth[1] = -s * 0.5 * sh

    dFdr[2] = 0.5 / s * sh * sth
    dFdth[2] = s * (0.5 * ch * sth + sh * cth)

    dFdr[3] = 0.5 / s * ch * sth
    dFdth[3] = s * (-0.5 * sh * sth + ch * cth)

    dFdx = dFdr * drdx + dFdth * dthdx
    dFdy = dFdr * drdy + dFdth * dthdy
    return F, dFdx, dFdy


def clip_segment_to_bbox(
    p0: np.ndarray,
    p1: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip segment p(t)=p0+t*(p1-p0), t∈[0,1] to axis-aligned bbox.

    Liang–Barsky clipping.
    """

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-18:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t

    q0 = np.array([x0 + u1 * dx, y0 + u1 * dy], dtype=float)
    q1 = np.array([x0 + u2 * dx, y0 + u2 * dy], dtype=float)
    if float(np.linalg.norm(q1 - q0)) <= 1e-14:
        return None
    return q0, q1
