"""Structured quad mesh generator."""

from __future__ import annotations
import numpy as np

def structured_quad_mesh(L: float, H: float, nx: int, ny: int):
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs], dtype=float)

    def nid(i, j):  # i along x, j along y
        return j * (nx + 1) + i

    elems = []
    for j in range(ny):
        for i in range(nx):
            n1 = nid(i, j)
            n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1)
            n4 = nid(i, j + 1)
            elems.append([n1, n2, n3, n4])
    return nodes, np.array(elems, dtype=int)


