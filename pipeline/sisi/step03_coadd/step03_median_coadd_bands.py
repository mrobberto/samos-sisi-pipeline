#!/usr/bin/env python3
"""
SISI / SAMOS Imaging — Step 03: coadd science frames per band (median stack).

Bands:
  r: 073..077
  i: 078..082
  z: 094..103

Inputs:
  SISI_REDUCED/02_biascorr/<NNN>.*_ampmatch.fits

Outputs:
  SISI_REDUCED/03_coadded/Coadd_<band>_median_<start>-<end>.fits
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits

import config  # must define SISI_ROOT and SISI_REDUCED (Path objects)
config.build_sisi_paths()


def nums(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


def read_primary(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32, copy=False)
        hdr = hdul[0].header
    return data, hdr


def find_processed_frame(in_dir: Path, n: int, band: str) -> Path:
    pattern = f"{n:03d}.sci_sloan-{band}_{config.SISI_TARGET}-{config.SISI_FIELD}_ampmatch.fits"
    matches = sorted(in_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No processed science file for {band} {n:03d}: {pattern}")

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous processed science file for {band} {n:03d}: {matches}")

    return matches[0]

def median_stack(paths: list[Path]) -> tuple[np.ndarray, fits.Header]:
    if len(paths) == 0:
        raise ValueError("No input files to stack.")

    cube = []
    first_hdr = None
    shape0 = None

    for p in paths:
        data, hdr = read_primary(p)
        if first_hdr is None:
            first_hdr = hdr
            shape0 = data.shape
        if data.shape != shape0:
            raise ValueError(f"Shape mismatch: {p.name} has {data.shape}, expected {shape0}")
        cube.append(data)

    cube = np.stack(cube, axis=0)  # (n, ny, nx)
    stacked = np.median(cube, axis=0).astype(np.float32)

    return stacked, first_hdr


def write_coadd(out_path: Path, data: np.ndarray, hdr0: fits.Header, inputs: list[Path], band: str):
    hdr = hdr0.copy()
    hdr["HISTORY"] = f"Median coadd for band {band}"
    hdr["HISTORY"] = f"NCOMBINE={len(inputs)}"
    # Keep HISTORY lines short (FITS cards)
    hdr["NCOMBINE"] = (len(inputs), "Number of frames combined")
    hdr["COMBMETH"] = ("MEDIAN", "Combination method")
    hdr["BAND"] = (band, "Coadd band label")

    for p in inputs:
        hdr["HISTORY"] = f"INPUT: {p.name}"

    fits.writeto(out_path, data, hdr, overwrite=True)


def main():
    in_dir = config.SISI_ST02_PREPROCESS
    out_dir = config.SISI_ST03_COADD
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = {
        band: list(frames)
        for band, frames in config.SISI_SCIENCE_FRAMES.items()
    }

    for band, frame_nums in bands.items():
        paths = []
        missing = []
        for n in frame_nums:
            p = find_processed_frame(in_dir, n, band)
            if p is None:
                missing.append(n)
            else:
                paths.append(p)

        if missing:
            raise FileNotFoundError(
                f"Missing processed inputs for band {band}: "
                + ", ".join(f"{m:03d}" for m in missing)
                + f"\nLooked in: {in_dir}"
            )

        print(f"\nBand {band}: stacking {len(paths)} frames")
        for p in paths:
            print("  ", p.name)

        coadd, hdr0 = median_stack(paths)

        out_name = f"Coadd_{band}_median_{frame_nums[0]:03d}-{frame_nums[-1]:03d}.fits"
        out_path = out_dir / out_name
        write_coadd(out_path, coadd, hdr0, paths, band)

        finite = np.isfinite(coadd)
        print(
            f"Wrote {out_path.name} | "
            f"median={np.median(coadd[finite]):.3f} std={np.std(coadd[finite]):.3f}"
        )

    print("\nDone. Output folder:", out_dir)


if __name__ == "__main__":
    main()
