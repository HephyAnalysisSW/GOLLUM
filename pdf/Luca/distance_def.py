"""
Helper functions to calculate the distance between a target PDF and a set of replicas.
"""

import numpy as np


def pdf_grid(pdf, pid, x_grid, Q):
    out = []
    for x in x_grid:
        out.append(pdf.xfxQ(pid, x, Q))
    return np.array(out)


def pdf_grid_allflav(pdf, flavours, x_grid, Q):
    out = np.array([])
    for pid in flavours:
        out = np.append(out, pdf_grid(pdf, pid, x_grid, Q))
    return out


def wmin_distance(
    pdf_target, center_pdf_grid, wmin_basis, flavours, x_grid, Q, dist_type=0
):
    Y = pdf_grid_allflav(pdf_target, flavours, x_grid, Q) - center_pdf_grid
    X = np.array(
        [
            pdf_grid_allflav(replica, flavours, x_grid, Q) - center_pdf_grid
            for replica in wmin_basis
        ]
    ).T

    if dist_type == 0:
        Sigma = np.identity(len(Y))

    elif dist_type == 1:
        Sigma = np.diag(np.abs(1 / (Y + center_pdf_grid)))

    elif dist_type == 2:
        Sigma = np.diag(np.abs(1 / (Y + center_pdf_grid) ** 2))

    elif dist_type == 3:
        X_grid = np.tile(x_grid, len(flavours))
        Sigma = np.diag(np.abs(X_grid / (Y + center_pdf_grid)))

    elif dist_type == 4:
        X_grid = np.tile(x_grid, len(flavours))
        Sigma = np.diag(np.abs(X_grid))

    w = np.linalg.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y
    distance = (Y - X @ w) @ Sigma @ (Y - X @ w)

    original = pdf_grid_allflav(pdf_target, flavours, x_grid, Q).reshape(
        len(flavours), len(x_grid)
    )
    reco = (center_pdf_grid + X @ w).reshape(len(flavours), len(x_grid))

    return original, reco, w, distance
