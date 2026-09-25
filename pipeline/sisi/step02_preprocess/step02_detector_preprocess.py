#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISI Imaging — Step02 preprocessing.

This step performs detector-level corrections immediately after the standard
orientation defined by Step00. The processing is entirely configuration driven,
allowing different observing modes and datasets to enable only the operations
required for their specific detector characteristics.

Step02 does not currently perform amplifier gain equalization.
Residual half-detector photometric corrections are applied later in Step05, after flat-fielding, when enabled by configuration.

Depending on the configuration, Step02 can perform:

  1. Science-frame pedestal-aware int16/u16 recovery
     - subtract independent amplifier pedestals
     - recover wrapped signed-int16 values into the full unsigned 16-bit
       dynamic range
     - leave the images in signal units without restoring a common pedestal

  2. Generic int16/u16 recovery
     - recover wrapped signed-int16 values without pedestal subtraction
     - intended for flat fields and other datasets that do not require the
       science pedestal correction

  3. Protected pass-through
     - frames requiring no detector correction are simply copied to the
       Step02 products while preserving provenance

Future detector-specific operations (e.g. amplifier gain equalization) may be
enabled through the configuration without changing the overall workflow.

No geometric operations are performed here. Image orientation, flips, and
rotations are assumed to have been completed in Step00.

Run:
===
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py --only 02

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

import config

# Keep compatibility with the existing SISI configuration shim, but do not require
# it if the loaded target config already defines all paths.
if hasattr(config, "build_sisi_paths"):
    config.build_sisi_paths()

WRITE_DEBUG_PRODUCTS = bool(getattr(config, "SISI_STEP02_WRITE_DEBUG", False))


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def cfg(name: str, default):
    """Read a configuration parameter with a local default."""
    return getattr(config, name, default)


def robust_sky(x: np.ndarray, method: str = "sigclip_median") -> float:
    """Robust scalar level from finite pixels."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")

    if method == "median":
        return float(np.median(x))

    if method == "sigclip_median":
        clipped = sigma_clip(x, sigma=3.0, maxiters=5)
        return float(np.median(clipped.compressed()))

    raise ValueError(f"Unknown sky estimator: {method}")


def frame_pattern(kind: str, n: int, band: str | None = None) -> str:
    """Build input filename pattern from config.SISI_FRAME_PATTERNS when present."""
    patterns = cfg("SISI_FRAME_PATTERNS", {})

    if kind == "science":
        template = patterns.get(
            "science",
            "{num:03d}.sci_sloan-{band}_{target}-{field}.fits",
        )
        return template.format(
            num=n,
            band=band,
            target=config.SISI_TARGET,
            field=config.SISI_FIELD,
        )

    if kind == "flat":
        flat_band = (band or "").split("_")[0]
        template = patterns.get("flat", "{num:03d}.flat_sloan-{flat_band}_Flat.fits")
        return template.format(num=n, band=band, flat_band=flat_band)

    if kind == "bias":
        template = patterns.get("bias", "{num:03d}.bias_*.fits")
        return template.format(num=n, band=band)

    raise ValueError(f"Unknown frame kind: {kind}")


def pick_input_file(fits_dir: Path, n: int, kind: str, band: str | None = None) -> Path:
    pattern = frame_pattern(kind=kind, n=n, band=band)
    matches = sorted(Path(fits_dir).glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No input file found for {kind} {n:03d}, band={band}, pattern={pattern}, "
            f"directory={fits_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous input file for {kind} {n:03d}, band={band}, pattern={pattern}:\n"
            + "\n".join(str(m) for m in matches)
        )

    return matches[0]


def read_primary_raw(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        raw = hdul[0].data
        hdr = hdul[0].header.copy()

    if raw is None:
        raise ValueError(f"Primary HDU has no image data: {path}")

    return np.asarray(raw), hdr


def signed_int16_to_unsigned_float(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Recover SISI data written as signed int16 but representing unsigned 16-bit ADU.

    Negative values are interpreted as wrapped values and shifted by +65536.
    This must be done before pedestal subtraction for the affected T00 science frames.
    """
    data = np.asarray(raw).astype(np.float32, copy=True)
    wrapped = np.isfinite(data) & (data < 0)
    nwrap = int(np.count_nonzero(wrapped))
    if nwrap:
        data[wrapped] += 65536.0
    return data, nwrap


def optional_generic_uint16_recovery(raw: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    """
    Generic read path for non-T00 science if configured.

    Flats are not sent here unless explicitly requested elsewhere.
    """
    if mode == "none":
        return np.asarray(raw).astype(np.float32, copy=True), 0
    if mode == "negative_only":
        return signed_int16_to_unsigned_float(raw)
    raise ValueError(f"Unsupported generic uint16 recovery mode: {mode}")


def get_pedestal_value(name: str, band: str | None, default: float) -> float:
    """
    Read a pedestal from config. Supports either scalar or dict-by-band values.
    """
    value = cfg(name, default)
    if isinstance(value, dict):
        if band in value:
            return float(value[band])
        if "default" in value:
            return float(value["default"])
        return float(default)
    return float(value)


# -----------------------------------------------------------------------------
# Core algorithms
# -----------------------------------------------------------------------------
def repair_science_ped_u16(raw: np.ndarray, band: str | None, method: str) -> tuple[np.ndarray, dict]:
    """
    Science pedestal-aware uint16 repair.

    Correct order for the pedestal-offset electronics mode:

      1. Split the Step00 image at config.SISI_AMP_SPLIT_COL.
      2. Subtract the local amplifier pedestal first:
            left_signal  = left_step00  - SISI_STEP02_SCIENCE_PED_LEFT   default 514 ADU
            right_signal = right_step00 - SISI_STEP02_SCIENCE_PED_RIGHT  default 540 ADU
      3. Pixels whose pedestal-subtracted signal is negative are interpreted as
         signed-int16 encodings of values just below zero, i.e. near saturation.
         Recover them by adding 65536 to the pedestal-subtracted signal.

         Example, left amplifier: x=513, pedestal=514
             signal_i16 = 513 - 514 = -1
             signal_u16 = -1 + 65536 = 65535

         Example, right amplifier: x=0, pedestal=540
             signal_i16 = 0 - 540 = -540
             signal_u16 = -540 + 65536 = 64996

      4. Do not restore a common pedestal and do not scale the amplifiers.

    This is deliberately pedestal-aware. A generic raw<0 unwrap before pedestal
    subtraction is not sufficient for this failure mode.
    """
    split_col = int(cfg("SISI_AMP_SPLIT_COL", 528))
    ped_l = get_pedestal_value("SISI_STEP02_SCIENCE_PED_LEFT", band, 514.0)
    ped_r = get_pedestal_value("SISI_STEP02_SCIENCE_PED_RIGHT", band, 540.0)

    data = np.asarray(raw).astype(np.float32, copy=True)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape={data.shape}")
    if not (0 < split_col < data.shape[1]):
        raise ValueError(f"Bad SISI_AMP_SPLIT_COL={split_col} for image shape={data.shape}")

    out = np.empty_like(data, dtype=np.float32)

    left_signal = data[:, :split_col] - ped_l
    right_signal = data[:, split_col:] - ped_r

    wrap_l = np.isfinite(left_signal) & (left_signal < 0)
    wrap_r = np.isfinite(right_signal) & (right_signal < 0)
    nwrap_l = int(np.count_nonzero(wrap_l))
    nwrap_r = int(np.count_nonzero(wrap_r))

    if nwrap_l:
        left_signal[wrap_l] += 65536.0
    if nwrap_r:
        right_signal[wrap_r] += 65536.0

    out[:, :split_col] = left_signal
    out[:, split_col:] = right_signal

    # Diagnostics only: these are not used to define the correction.
    edge_width = int(cfg("SISI_STEP02_SCIENCE_PED_EDGE_WIDTH", 30))
    sky_margin = int(cfg("SISI_STEP02_SCIENCE_SKY_MARGIN", 60))

    left_edge = data[:, :edge_width]
    right_edge = data[:, -edge_width:]
    left_sky = data[:, sky_margin:max(sky_margin + 1, split_col - edge_width)]
    right_sky = data[:, min(data.shape[1] - sky_margin, split_col + edge_width):data.shape[1] - sky_margin]

    meta = {
        "nwrap": nwrap_l + nwrap_r,
        "nwrap_l": nwrap_l,
        "nwrap_r": nwrap_r,
        "ped_l": ped_l,
        "ped_r": ped_r,
        "split_col": split_col,
        "edge_l": robust_sky(left_edge.ravel(), method=method),
        "edge_r": robust_sky(right_edge.ravel(), method=method),
        "sky_l_raw": robust_sky(left_sky.ravel(), method=method),
        "sky_r_raw": robust_sky(right_sky.ravel(), method=method),
        "nneg_raw": int(np.count_nonzero(np.asarray(raw) < 0)),
        "nneg_after_ped_l": nwrap_l,
        "nneg_after_ped_r": nwrap_r,
        "nneg_out": int(np.count_nonzero(np.isfinite(out) & (out < 0))),
        "min_raw_step00": float(np.nanmin(data)),
        "max_raw_step00": float(np.nanmax(data)),
        "min_out": float(np.nanmin(out)),
        "max_out": float(np.nanmax(out)),
        "med_left_out": float(np.nanmedian(out[:, :split_col])),
        "med_right_out": float(np.nanmedian(out[:, split_col:])),
    }
    return out, meta


def passthrough_frame(raw: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """
    Protected output path for flats and non-special science frames.

    Protected means: do not subtract T00 pedestals and do not do amplifier
    matching. It does *not* necessarily mean byte-for-byte pass-through.
    Flats can still need the generic signed-int16 -> unsigned-16 recovery
    because saturated/near-saturated flat pixels may be stored as negative
    int16 values.
    """
    if kind == "science":
        mode = str(cfg("SISI_STEP02_OTHER_SCIENCE_U16_MODE", "none"))
    elif kind == "flat":
        # Flats may include saturated pixels stored as signed int16 negatives.
        # If enabled, recover them by adding 65536 to raw<0 pixels only.
        # No pedestal subtraction, no pedestal-aware correction, no amp scaling.
        mode = "negative_only" if bool(cfg("SISI_STEP02_FLAT_U16_FIX", True)) else "none"
    else:
        mode = "none"

    data, nwrap = optional_generic_uint16_recovery(raw, mode=mode)
    meta = {
        "nwrap": nwrap,
        "u16mode": mode,
        "ped_l": 0.0,
        "ped_r": 0.0,
        "split_col": int(cfg("SISI_AMP_SPLIT_COL", 528)),
        "nneg_raw": int(np.count_nonzero(np.asarray(raw) < 0)),
        "nneg_out": int(np.count_nonzero(np.isfinite(data) & (data < 0))),
        "min_out": float(np.nanmin(data)),
        "max_out": float(np.nanmax(data)),
    }
    return data.astype(np.float32, copy=False), meta


def should_apply_science_ped_u16_fix(kind: str) -> bool:
    """Single gate for science-frame pedestal-aware int16/u16 repair.

    This is intentionally dataset-neutral: configs choose which SISI_FIELD values
    receive the repair through SISI_STEP02_SCIENCE_PED_U16_FIELDS.
    """
    enabled = bool(cfg("SISI_STEP02_SCIENCE_PED_U16_FIX", False))
    allowed_fields = cfg("SISI_STEP02_SCIENCE_PED_U16_FIELDS", ())
    return enabled and kind == "science" and str(config.SISI_FIELD) in set(allowed_fields)

def write_step02_output(
    out_path: Path,
    data: np.ndarray,
    hdr: fits.Header,
    kind: str,
    band: str | None,
    repaired: bool,
    meta: dict,
) -> None:
    hdr_out = hdr.copy()
    hdr_out["ST02KIND"] = (kind, "Step02 frame class")
    hdr_out["ST02BAND"] = (str(band), "Step02 band / flat key")
    hdr_out["SISIFLD"] = (str(config.SISI_FIELD), "SISI slit/acquisition field")
    hdr_out["PEDU16"] = (bool(repaired), "Science pedestal-aware uint16 repair applied")
    hdr_out["I16FIX"] = (int(meta.get("nwrap", 0)) > 0, "Signed-int16 wrapped pixels recovered")
    hdr_out["I16NPIX"] = (int(meta.get("nwrap", 0)), "Pedestal-subtracted pixels shifted by +65536")
    hdr_out["I16PED"] = (bool(repaired), "Unsigned recovery done after pedestal subtraction")
    hdr_out["U16MODE"] = (str(meta.get("u16mode", "pedestal_aware" if repaired else "none")), "Step02 uint16 recovery mode")
    hdr_out["I16NPL"] = (int(meta.get("nwrap_l", 0)), "Left pixels shifted by +65536 after pedestal subtraction")
    hdr_out["I16NPR"] = (int(meta.get("nwrap_r", 0)), "Right pixels shifted by +65536 after pedestal subtraction")
    hdr_out["PEDSUB"] = (bool(repaired), "Independent amp pedestals subtracted")
    hdr_out["PED_L"] = (float(meta.get("ped_l", 0.0)), "Left amp pedestal subtracted")
    hdr_out["PED_R"] = (float(meta.get("ped_r", 0.0)), "Right amp pedestal subtracted")
    hdr_out["PEDREST"] = (False, "No common pedestal restored")
    hdr_out["AMPMATCH"] = (False, "No multiplicative amp scaling in Step02")
    hdr_out["GAINR2L"] = (1.0, "No right-to-left gain scaling")
    hdr_out["SPLITCOL"] = (int(meta.get("split_col", cfg("SISI_AMP_SPLIT_COL", 528))), "Amplifier split column")
    hdr_out["NNEGRAW"] = (int(meta.get("nneg_raw", 0)), "Negative pixels in raw input array")
    hdr_out["NNEGOUT"] = (int(meta.get("nneg_out", 0)), "Negative pixels after Step02 output")

    if repaired:
        hdr_out["EDGE_L"] = (float(meta.get("edge_l", np.nan)), "Diagnostic left edge level before subtraction")
        hdr_out["EDGE_R"] = (float(meta.get("edge_r", np.nan)), "Diagnostic right edge level before subtraction")
        hdr_out["SKY_L"] = (float(meta.get("sky_l_raw", np.nan)), "Diagnostic left sky before subtraction")
        hdr_out["SKY_R"] = (float(meta.get("sky_r_raw", np.nan)), "Diagnostic right sky before subtraction")
        hdr_out["HISTORY"] = "Step02 science repair: subtract amp pedestals, then pedestal-aware int16 unwrap."
    else:
        if kind == "flat" and int(meta.get("nwrap", 0)) > 0:
            hdr_out["HISTORY"] = "Step02 protected flat: generic signed-int16 unwrap only; no pedestal/amp repair."
        else:
            hdr_out["HISTORY"] = "Step02 protected pass-through: no science pedestal/u16 repair applied."

    fits.writeto(out_path, data.astype(np.float32, copy=False), hdr_out, overwrite=True)


def output_name(in_path: Path) -> str:
    suffix = str(cfg("SISI_STEP02_OUTPUT_SUFFIX", "_ampmatch"))
    return f"{in_path.stem}{suffix}.fits"


def process_frame_list(
    frame_nums: Iterable[int],
    fits_dir: Path,
    out_dir: Path,
    method: str = "sigclip_median",
    kind: str | None = None,
    band: str | None = None,
) -> None:
    if kind is None:
        raise ValueError("process_frame_list requires kind='science' or kind='flat'")

    fits_dir = Path(fits_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in sorted(frame_nums):
        in_path = pick_input_file(fits_dir, n, kind=kind, band=band)
        raw, hdr = read_primary_raw(in_path)

        repaired = should_apply_science_ped_u16_fix(kind)
        if repaired:
            out, meta = repair_science_ped_u16(raw, band=band, method=method)
            action = "SCI_PED_U16"
        else:
            out, meta = passthrough_frame(raw, kind=kind)
            action = "PROTECTED_PASS"

        out_path = out_dir / output_name(in_path)
        write_step02_output(out_path, out, hdr, kind, band, repaired, meta)

        print(
            f"{action:14s} {in_path.name} -> {out_path.name}  "
            f"I16NPIX={meta.get('nwrap', 0):7d}  "
            f"PED_L={meta.get('ped_l', 0.0):7.1f} PED_R={meta.get('ped_r', 0.0):7.1f}  "
            f"NNEG_RAW={meta.get('nneg_raw', 0):7d} NNEG_OUT={meta.get('nneg_out', 0):7d}  "
            f"OUT_MIN/MAX=({meta.get('min_out', np.nan):.1f},{meta.get('max_out', np.nan):.1f})"
        )

        if repaired:
            print(
                f"    diagnostics: EDGE_L={meta['edge_l']:.1f} EDGE_R={meta['edge_r']:.1f}  "
                f"SKY_L_RAW={meta['sky_l_raw']:.1f} SKY_R_RAW={meta['sky_r_raw']:.1f}  "
                f"I16_L/R=({meta.get('nwrap_l', 0)},{meta.get('nwrap_r', 0)})  "
                f"MED_OUT_L/R=({meta['med_left_out']:.1f},{meta['med_right_out']:.1f})"
            )

        if WRITE_DEBUG_PRODUCTS and repaired:
            debug_dir = out_dir.parent / f"{out_dir.name}_debug_step00_before_t00_repair"
            debug_dir.mkdir(parents=True, exist_ok=True)
            fits.writeto(debug_dir / f"{in_path.stem}_step00.fits", raw, hdr, overwrite=True)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    method = str(cfg("SISI_STEP02_SKY_METHOD", "sigclip_median"))

    print("\n=== Step02: science night frames ===")
    print(
        f"Config gate: SISI_FIELD={config.SISI_FIELD!r}; "
        f"science ped/u16 fix enabled={bool(cfg('SISI_STEP02_SCIENCE_PED_U16_FIX', False))}; "
        f"allowed fields={tuple(cfg('SISI_STEP02_SCIENCE_PED_U16_FIELDS', ()))}; "
        f"flat u16 fix={bool(cfg('SISI_STEP02_FLAT_U16_FIX', True))}"
    )

    for band, frames in config.SISI_SCIENCE_FRAMES.items():
        print(f"\n--- Science band {band} ---")
        process_frame_list(
            frames,
            config.SISI_ST00_ORIENT,
            config.SISI_ST02_PREPROCESS,
            method=method,
            kind="science",
            band=band,
        )

    if hasattr(config, "SISI_FLAT_FRAMES"):
        print("\n=== Step02: flat night frames ===")
        print("Flat frames are protected from science pedestal/amp repair; flat uint16 unwrap is config-controlled.")

        # Prefer a dedicated flat Step00 directory if the target config defines one.
        flat_fits_dir = Path(cfg("SISI_ST00_ORIENT_FLATS", cfg("SISI_ST00_ORIENT", config.SISI_ST00_ORIENT)))

        for flat_key, frames in config.SISI_FLAT_FRAMES.items():
            print(f"\n--- Flat set {flat_key} ---")
            process_frame_list(
                frames,
                flat_fits_dir,
                config.SISI_ST02_PREPROCESS_FLATS,
                method=method,
                kind="flat",
                band=flat_key,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
