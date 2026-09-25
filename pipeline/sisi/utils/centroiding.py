# pipeline/sisi/utils/centroiding.py
from __future__ import annotations

import numpy as np
from photutils.centroids import centroid_com


def peak_centered_centroid(
    data,
    x0,
    y0,
    search_box: int = 11,
    centroid_box: int = 7,
    max_shift: float = 6.0,
):
    """
    Peak-centered local centroid for SISI forced photometry.

    Start from a WCS/catalog position, find the local background-subtracted
    peak in a search box, then compute the centroid only in a smaller box
    around that peak. This is more robust than a center-of-gravity centroid
    over the full search box.
    """
    ny, nx = data.shape
    hs = search_box // 2

    ix = int(round(x0))
    iy = int(round(y0))

    if ix - hs < 0 or ix + hs >= nx or iy - hs < 0 or iy + hs >= ny:
        return np.nan, np.nan, np.nan

    cut = data[iy-hs:iy+hs+1, ix-hs:ix+hs+1].copy()
    if not np.isfinite(cut).any():
        return np.nan, np.nan, np.nan

    med = np.nanmedian(cut)
    work = cut - med
    work[~np.isfinite(work)] = 0.0

    py, px = np.unravel_index(np.nanargmax(work), work.shape)

    hc = centroid_box // 2
    y1 = max(py - hc, 0)
    y2 = min(py + hc + 1, work.shape[0])
    x1 = max(px - hc, 0)
    x2 = min(px + hc + 1, work.shape[1])

    sub = work[y1:y2, x1:x2].copy()
    sub[sub < 0] = 0.0

    if np.nansum(sub) <= 0:
        return np.nan, np.nan, np.nan

    cy, cx = centroid_com(sub)

    x = ix - hs + x1 + cx
    y = iy - hs + y1 + cy
    shift = float(np.hypot(x - x0, y - y0))

    if shift > max_shift:
        return np.nan, np.nan, shift

    return float(x), float(y), shift