#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:44:24 2026

@author: robberto

SISI / SAMOS Imaging — Step 01: build SuperBias from bias frames.

Combines:
  004.bias_Westerlund1-T00.fits  ...  023.bias_Westerlund1-T00.fits

Inputs:
  SISI_root / fits / <files>

Outputs:
  SISI_reduced / 01_bias / SuperBias_Westerlund1-T00_004-023.fits

Relies on config.py for directory structure.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

import config  # must define SISI_root and SISI_reduced (Path objects)


def build_file_list(start: int, end: int, suffix: str = "bias") -> list[Path]:
    fits_dir = config.SISI_ST00_ORIENT
    files = []

    for i in range(start, end + 1):
        matches = sorted(fits_dir.glob(f"{i:03d}.{suffix}_*.fits"))
        if not matches:
            matches = sorted(fits_dir.glob(f"{i:03d}.{suffix}*.fits"))
        if matches:
            files.append(matches[0])
        else:
            files.append(fits_dir / f"{i:03d}.{suffix}_MISSING.fits")

    return files

def read_fits_data(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32, copy=False)
        hdr = hdul[0].header
    return data, hdr


def main():
    # ---- configure this step ----
    start, end = 4, 23
    suffix = "bias"  # yields 004.bias_Westerlund1-T00.fits, etc.

    out_dir = Path(config.SISI_REDUCED) / "01_bias"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"SuperBias_{config.SISI_TARGET}_{config.SISI_FIELD}_{start:03d}-{end:03d}.fits"
    out_path = out_dir / out_name

    # ---- gather inputs ----
    files = build_file_list(start, end, suffix)

    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input bias frames:\n" + "\n".join(str(p) for p in missing)
        )

    print(f"Reading {len(files)} bias frames from: {files[0].parent}")
    cube = []
    first_hdr = None
    for p in files:
        data, hdr = read_fits_data(p)
        if first_hdr is None:
            first_hdr = hdr
        cube.append(data)

    cube = np.stack(cube, axis=0)  # (n, ny, nx)
    print("Data cube shape:", cube.shape, "dtype:", cube.dtype)

    # ---- robust combine: sigma-clipped mean per pixel ----
    clipped = sigma_clip(cube, sigma=4.0, maxiters=5, axis=0)
    superbias = np.ma.mean(clipped, axis=0).filled(np.nan).astype(np.float32)

    # ---- write output ----
    hdr_out = first_hdr.copy()
    hdr_out["HISTORY"] = f"SuperBias = sigma-clipped mean of {len(files)} bias frames"
    hdr_out["HISTORY"] = f"Inputs: {files[0].name} .. {files[-1].name}"
    hdr_out["NCOMBINE"] = (len(files), "Number of frames combined")
    hdr_out["COMBMETH"] = ("SIGCLIP_MEAN", "Combination method")

    fits.writeto(out_path, superbias, hdr_out, overwrite=True)
    print("Wrote:", out_path)

    # Optional quick stats
    finite = np.isfinite(superbias)
    print(
        "SuperBias stats (finite pixels):",
        f"median={np.median(superbias[finite]):.3f}",
        f"std={np.std(superbias[finite]):.3f}",
        f"min={np.min(superbias[finite]):.3f}",
        f"max={np.max(superbias[finite]):.3f}",
    )


if __name__ == "__main__":
    main()
