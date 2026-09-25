#!/usr/bin/env python3
"""
run
===
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py --only 00
"""


from pathlib import Path
import numpy as np
from astropy.io import fits
import config

config.build_sisi_paths()

IN_DIR = Path(config.SISI_RAW)
OUT_DIR = Path(config.SISI_ST00_ORIENT)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = "*.fits"

def collect_inputs():
    rows = []

    # Science-night bias frames
    for n in config.SISI_BIAS_FRAMES:
        rows.append((int(n), Path(config.SISI_RAW), "bias/science-night"))

    # Science frames
    for band, frames in config.SISI_SCIENCE_FRAMES.items():
        for n in frames:
            rows.append((int(n), Path(config.SISI_RAW), f"science {band}"))

    # Flat-night frames
    for band, frames in config.SISI_FLAT_FRAMES.items():
        for n in frames:
            rows.append((int(n), Path(config.SISI_FLAT_ROOT) / "fits", f"flat {band}"))

    return rows

def expand_frame_numbers() -> set[int]:
    nums = set()

    if hasattr(config, "SISI_BIAS_FRAMES"):
        nums.update(config.SISI_BIAS_FRAMES)

    if hasattr(config, "SISI_SCIENCE_FRAMES"):
        for frames in config.SISI_SCIENCE_FRAMES.values():
            nums.update(frames)

    if hasattr(config, "SISI_FLAT_FRAMES"):
        for frames in config.SISI_FLAT_FRAMES.values():
            nums.update(frames)

    return set(int(n) for n in nums)

wanted = expand_frame_numbers()
print(f"Step00 will orient {len(wanted)} configured frames from {IN_DIR}")

for old in OUT_DIR.glob("*.fits"):
    old.unlink()
print("Cleared old Step00 products:", OUT_DIR)

for n, in_dir, label in collect_inputs():
    matches = sorted(in_dir.glob(f"{n:03d}*.fits"))
    if not matches:
        print(f"SKIP {n:03d} {label}: no matching FITS in {in_dir}")
        continue

    f = matches[0]
    with fits.open(f, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()

    if data is None or data.ndim != 2:
        print("SKIP:", f.name)
        continue

    # Fix raw SISI orientation once, at the start of the pipeline.
    data_out = np.fliplr(data).astype(np.float32)

    hdr["ORIENTED"] = (True, "Raw SISI frame flipped to standard orientation")
    hdr["FLIPX"] = (True, "X mirror applied in Step00")
    hdr["HISTORY"] = "Step00: np.fliplr applied to raw SISI image"

    out = OUT_DIR / f.name
    fits.writeto(out, data_out, hdr, overwrite=True)
    print("Wrote:", out)



print("Done. Output folder:", OUT_DIR)