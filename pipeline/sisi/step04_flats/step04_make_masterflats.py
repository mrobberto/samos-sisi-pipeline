#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISI imaging — Step04: master-flat construction

This step reads the Step02-processed flat frames, performs only optional
bookkeeping-level preprocessing, combines the flats, and writes normalized
master flats for the science bands requested by the active configuration.

Production philosophy
---------------------
Step04 should preserve the multiplicative detector response measured by the
flats, including any left/right amplifier gain difference.  Therefore the
production defaults are:

  * no additive amplifier equalization;
  * no multiplicative half matching;
  * median-combine ordinary flat groups;
  * for i band, when both i_025s and i_05s groups exist and are usable, build a
    differential flat from median(i_05s) - median(i_025s), then normalize.

The i-band differential flat is preferred because it cancels residual additive
terms that are common to both exposure levels while preserving the multiplicative
response that Step06 should divide out of the science frames.

The r and z bands normally use an ordinary median master flat, because only one
usable exposure level may be available.

Inputs
------
  config.SISI_ST02_PREPROCESS_FLATS, or fallback:
  config.SISI_REDUCED / "02_preprocess_flats"

Expected input names currently follow the Step02 suffix convention, e.g.
  <NNN>.flat_sloan-<band>_Flat_ampmatch.fits
where the suffix name may be historical even when Step02 amp matching is off.

Outputs
-------
  config.SISI_ST04_FLAT_PREPROC / <input_stem>_flatprep.fits
  config.SISI_ST04_MASTERFLATS / MasterFlat_<band>_norm.fits
"""

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

import config
config.build_sisi_paths()

# -----------------------------------------------------------------------------
# User-tunable knobs
# -----------------------------------------------------------------------------

split_col = config.SISI_AMP_SPLIT_COL
SUPERBIAS_DIR = config.SISI_ST01_BIAS

# Production defaults: keep detector/amplifier response in the flat.
DO_AMP_EQUALIZE = bool(getattr(config, "SISI_STEP04_DO_AMP_EQUALIZE", False)) # additive right-half shift; diagnostic only
DO_ILLUM_MATCH_HALVES = bool(getattr(config, "SISI_STEP04_DO_ILLUM_MATCH_HALVES", False)) # multiplicative right-half scaling; diagnostic only
DO_SUPERBIAS_SUBTRACT = bool(getattr(config, "SISI_STEP04_DO_SUPERBIAS_SUBTRACT", False)) # Step02 flats should already be bias/ped corrected
# Prefer differential i-band flat if i_025s and i_05s are both present?
USE_I_DIFFERENTIAL_FLAT = bool(getattr(config, "SISI_STEP04_USE_I_DIFFERENTIAL_FLAT", False))
MIN_FLATS_PER_GROUP = int(getattr(config, "SISI_STEP04_MIN_FLATS_PER_GROUP", 3))


# Saturation heuristic.  With max_saturated_fraction=1.0, this only rejects fully
# saturated/all-bad frames.  Tighten to e.g. 1e-4 if desired.
sat_thresh_frac_of_fullscale = 0.98
max_saturated_fraction = 1.0

# -----------------------------------------------------------------------------


def read_primary(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32, copy=False)
        hdr = hdul[0].header.copy()
    return data, hdr


def newest_superbias(superbias_dir: Path) -> Path:
    cands = sorted(superbias_dir.glob("SuperBias*.fits"))
    if not cands:
        raise FileNotFoundError(f"No SuperBias*.fits found in {superbias_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def robust_median(x: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    xc = sigma_clip(x, sigma=sigma, maxiters=maxiters)
    return float(np.median(xc.compressed()))


def amp_equalize_additive(img: np.ndarray, split: int) -> tuple[np.ndarray, float, float, float]:
    """Diagnostic-only additive right-half shift to match left median."""
    left = img[:, :split]
    right = img[:, split:]
    sky_l = robust_median(left.ravel())
    sky_r = robust_median(right.ravel())
    delta = sky_l - sky_r
    out = img.astype(np.float32, copy=True)
    out[:, split:] += delta
    return out, sky_l, sky_r, delta


def match_illum_halves_multiplicative(
    img: np.ndarray,
    split: int,
    ped_left: float | None = None,
    ped_right: float | None = None,
    thresh: float = 200.0,
) -> tuple[np.ndarray, float, float, float]:
    """Diagnostic-only right-half scaling to match illuminated left median."""
    out = img.astype(np.float32, copy=True)
    left = out[:, :split]
    right = out[:, split:]

    if ped_left is None or not np.isfinite(ped_left):
        ped_left = robust_median(left.ravel())
    if ped_right is None or not np.isfinite(ped_right):
        ped_right = robust_median(right.ravel())

    mask_l = left > (ped_left + thresh)
    mask_r = right > (ped_right + thresh)

    med_l = robust_median(left[mask_l])
    med_r = robust_median(right[mask_r])

    if not np.isfinite(med_l) or not np.isfinite(med_r) or med_r == 0.0:
        scale_r = 1.0
    else:
        scale_r = med_l / med_r
        out[:, split:] *= scale_r

    return out, med_l, med_r, scale_r


def estimate_fullscale(hdr: fits.Header) -> float:
    for key in ("SATURATE", "SATLEVEL", "SATSIG", "DATAMAX"):
        if key in hdr:
            try:
                return float(hdr[key])
            except Exception:
                pass
    return 65535.0


def is_saturated(data: np.ndarray, fullscale: float) -> tuple[bool, float]:
    thresh = sat_thresh_frac_of_fullscale * fullscale
    finite = np.isfinite(data)
    if finite.sum() == 0:
        return True, 1.0
    frac = float(np.mean(data[finite] >= thresh))
    return (frac > max_saturated_fraction), frac


def median_stack(paths: list[Path]) -> tuple[np.ndarray, fits.Header]:
    if len(paths) == 0:
        raise ValueError("median_stack received an empty path list")

    cube = []
    hdr0 = None
    shape0 = None
    for p in paths:
        d, h = read_primary(p)
        if hdr0 is None:
            hdr0 = h
            shape0 = d.shape
        if d.shape != shape0:
            raise ValueError(f"Shape mismatch: {p.name} {d.shape} vs {shape0}")
        cube.append(d)

    cube = np.stack(cube, axis=0)
    return np.median(cube, axis=0).astype(np.float32), hdr0


def normalize_to_one(flat: np.ndarray) -> tuple[np.ndarray, float]:
    med = robust_median(flat.ravel())
    if not np.isfinite(med) or med == 0.0:
        raise ValueError("Cannot normalize flat: median is non-finite or zero.")
    return (flat / med).astype(np.float32), med


def input_dir() -> Path:
    return Path(getattr(config, "SISI_ST02_PREPROCESS_FLATS", config.SISI_REDUCED / "02_preprocess_flats"))


def step02_suffix() -> str:
    return str(getattr(config, "SISI_STEP02_OUTPUT_SUFFIX", "_ampmatch"))


def pick_file(in_dir: Path, n: int, label: str) -> Path:
    flat_band = label.split("_")[0]  # i_025s -> i
    suffix = step02_suffix()

    patterns = [
        f"{n:03d}.flat_sloan-{flat_band}_Flat{suffix}.fits",
        f"{n:03d}.flat_sloan-{flat_band}_Flat*.fits",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches = sorted(in_dir.glob(pattern))
        if matches:
            break

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No Step02 processed flat for {label} frame {n:03d} in {in_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous processed flat for {label} {n:03d}: {matches}")
    return matches[0]


def hdr_float(x, bad: float = -9999.0) -> float:
    try:
        x = float(x)
    except Exception:
        return bad
    return x if np.isfinite(x) else bad


def preprocess_flat(
    in_path: Path,
    out_path: Path,
    superbias: np.ndarray | None = None,
    superbias_path: Path | None = None,
) -> tuple[bool, float]:
    img, hdr = read_primary(in_path)
    out = img.astype(np.float32, copy=True)

    illum_l = illum_r = scale_r = np.nan
    sky_l = sky_r = delta = np.nan

    if DO_SUPERBIAS_SUBTRACT:
        if superbias is None:
            raise RuntimeError("DO_SUPERBIAS_SUBTRACT=True but superbias was not loaded")
        if out.shape != superbias.shape:
            raise ValueError(f"{in_path.name}: shape {out.shape} != superbias {superbias.shape}")
        out = (out - superbias).astype(np.float32)

    if DO_ILLUM_MATCH_HALVES:
        out, illum_l, illum_r, scale_r = match_illum_halves_multiplicative(
            out,
            split_col,
            ped_left=hdr.get("SKY_L", np.nan),
            ped_right=hdr.get("SKY_R", np.nan),
            thresh=200.0,
        )

    if DO_AMP_EQUALIZE:
        out, sky_l, sky_r, delta = amp_equalize_additive(out, split_col)

    fullscale = estimate_fullscale(hdr)
    sat, satfrac = is_saturated(out, fullscale)

    hdr_out = hdr.copy()
    hdr_out["HISTORY"] = "Step04 flat preprocessing"
    hdr_out["AMPADD"] = (bool(DO_AMP_EQUALIZE), "Diagnostic additive amp equalization applied")
    hdr_out["ILLMATCH"] = (bool(DO_ILLUM_MATCH_HALVES), "Diagnostic multiplicative half match applied")
    hdr_out["SUPERBS"] = (bool(DO_SUPERBIAS_SUBTRACT), "Additional superbias subtraction applied")
    hdr_out["SPLITCOL"] = (int(split_col), "Amplifier split column, Python convention")

    if DO_SUPERBIAS_SUBTRACT and superbias_path is not None:
        hdr_out["BIASFILE"] = (superbias_path.name, "Additional superbias used")

    hdr_out["SKY_L04"] = (hdr_float(sky_l), "Step04 diagnostic left median")
    hdr_out["SKY_R04"] = (hdr_float(sky_r), "Step04 diagnostic right median")
    hdr_out["DELR04"] = (hdr_float(delta), "Step04 additive term applied to right half")
    hdr_out["ILL_L04"] = (hdr_float(illum_l), "Step04 diagnostic illuminated left median")
    hdr_out["ILL_R04"] = (hdr_float(illum_r), "Step04 diagnostic illuminated right median")
    hdr_out["ILLSC04"] = (hdr_float(scale_r), "Step04 diagnostic scale applied to right half")
    hdr_out["SATFRAC"] = (float(satfrac), "Fraction pixels above saturation threshold")
    hdr_out["SATFLAG"] = (bool(sat), "Saturation flag")

    fits.writeto(out_path, out.astype(np.float32), hdr_out, overwrite=True)
    return sat, satfrac


def build_master_for_band(
    band: str,
    paths: list[Path],
    usable: dict[str, list[Path]],
) -> tuple[np.ndarray, fits.Header, str, list[Path]]:
    """Return unnormalized master image, header seed, method name, input list."""

    if (
        band == "i"
        and USE_I_DIFFERENTIAL_FLAT
        and len(usable.get("i_025s", [])) >= MIN_FLATS_PER_GROUP
        and len(usable.get("i_05s", [])) >= MIN_FLATS_PER_GROUP
    ):
        paths025 = usable["i_025s"]
        paths05 = usable["i_05s"]
        flat025, hdr0 = median_stack(paths025)
        flat05, _ = median_stack(paths05)
        flat = (flat05 - flat025).astype(np.float32)
        method = "DIFF_I_05S_MINUS_I_025S"
        used = paths05 + paths025

        print("i differential flat:")
        print(f"  N i_05s  = {len(paths05)}")
        print(f"  N i_025s = {len(paths025)}")
        print(f"  median   = {np.nanmedian(flat):.6g}")
        print(f"  min/max  = {np.nanmin(flat):.6g} / {np.nanmax(flat):.6g}")
        return flat, hdr0, method, used

    if len(paths) < MIN_FLATS_PER_GROUP:
        raise RuntimeError(f"Not enough usable flats for band {band}: {len(paths)}")

    flat, hdr0 = median_stack(paths)
    return flat, hdr0, "MEDIAN", paths


def main() -> None:
    in_dir = input_dir()
    preproc_dir = Path(config.SISI_ST04_FLAT_PREPROC)
    master_dir = Path(config.SISI_ST04_MASTERFLATS)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)

    superbias = None
    superbias_path = None
    if DO_SUPERBIAS_SUBTRACT:
        superbias_path = newest_superbias(SUPERBIAS_DIR)
        superbias, _ = read_primary(superbias_path)
        print("Using additional superbias:", superbias_path)

    groups = {label: list(frames) for label, frames in config.SISI_FLAT_FRAMES.items()}
    usable: dict[str, list[Path]] = {k: [] for k in groups}

    print("Input dir:", in_dir)
    print("Flat preproc dir:", preproc_dir)
    print("Master flat dir:", master_dir)
    print("Production switches:")
    print(f"  DO_ILLUM_MATCH_HALVES = {DO_ILLUM_MATCH_HALVES}")
    print(f"  DO_AMP_EQUALIZE       = {DO_AMP_EQUALIZE}")
    print(f"  USE_I_DIFFERENTIAL    = {USE_I_DIFFERENTIAL_FLAT}")

    for label, frame_nums in groups.items():
        if len(frame_nums) == 0:
            continue
        print(f"\n--- Preprocessing group {label} ({frame_nums[0]:03d}-{frame_nums[-1]:03d}) ---")
        for n in frame_nums:
            p = pick_file(in_dir, n, label)
            out_path = preproc_dir / f"{p.stem}_flatprep.fits"
            sat, satfrac = preprocess_flat(p, out_path, superbias, superbias_path)

            msg = f"{p.name} -> {out_path.name} | satfrac={satfrac:.2e}"
            if sat:
                print("SATURATED:", msg)
            else:
                print("OK       :", msg)
                usable[label].append(out_path)

    masters: dict[str, tuple[list[Path], str]] = {}
    for master_band, group_labels in config.SISI_FLAT_MASTER_BANDS.items():
        paths: list[Path] = []
        for label in group_labels:
            paths.extend(usable.get(label, []))
        masters[master_band] = (paths, f"Using configured flat groups: {', '.join(group_labels)}")

    for band, (paths, note) in masters.items():
        print(f"\n=== MasterFlat {band} ===")
        print(note)

        flat, hdr0, method, used_paths = build_master_for_band(band, paths, usable)
        if len(used_paths) < MIN_FLATS_PER_GROUP:
            raise RuntimeError(f"Not enough usable flats for band {band}: {len(used_paths)}")

        for p in used_paths:
            print("  ", p.name)

        flat_norm, norm_med = normalize_to_one(flat)

        out_path = master_dir / f"MasterFlat_{band}_norm.fits"
        hdr = hdr0.copy()
        hdr["HISTORY"] = f"Step04 master flat band={band}"
        hdr["HISTORY"] = "Production default preserves amplifier gain response"
        hdr["HISTORY"] = "Inputs are Step04 *_flatprep.fits from Step02 flats"
        hdr["NCOMBINE"] = (len(used_paths), "Number of input flats used")
        hdr["COMBMETH"] = (method, "Combination method")
        hdr["NORMMED"] = (float(norm_med), "Median used for normalization")
        hdr["NORM"] = (True, "Flat normalized to median=1")
        hdr["AMPADD"] = (bool(DO_AMP_EQUALIZE), "Diagnostic additive amp equalization applied")
        hdr["ILLMATCH"] = (bool(DO_ILLUM_MATCH_HALVES), "Diagnostic multiplicative half match applied")
        hdr["SUPERBS"] = (bool(DO_SUPERBIAS_SUBTRACT), "Additional superbias subtraction applied")
        hdr["DIFFI"] = (bool(method.startswith("DIFF_I")), "i flat built from 0.5s - 0.25s medians")
        for p in used_paths:
            hdr["HISTORY"] = f"INPUT: {p.name}"

        fits.writeto(out_path, flat_norm.astype(np.float32), hdr, overwrite=True)
        print("Wrote:", out_path)
        print(f"  method  = {method}")
        print(f"  normmed = {norm_med:.6g}")

    print("\nDone.")
    print("Preprocessed flats:", preproc_dir)
    print("Master flats:", master_dir)


if __name__ == "__main__":
    main()
