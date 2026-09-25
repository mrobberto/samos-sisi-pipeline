#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step07c -- propagate the final SISI WCS to flat-fielded science images.

This step does not solve a new astrometric model.  It copies the adopted
master WCS from the reference frame and, by default, attaches the frozen
instrument-level SIP distortion calibration.  The final operational science
products are written as ``*_wcs.fits`` and are the files to be used by all
downstream steps.

Inputs
------
- Flat-fielded science images from ``config.SISI_ST05_FLATFIELD`` matching
  ``*_ff.fits``.
- ``config.SISI_WCS_REF_FINAL``: preferred immutable final adopted WCS reference.
- ``config.SISI_WCS_REF_FRAME`` and ``config.SISI_WCS_REF_MASTER``: fallbacks.
  The reference supplies the absolute linear TAN solution; SIP is always
  reattached from the frozen JSON calibration.
- ``config.SISI_SIP_CALIBRATION``: frozen SIP calibration JSON, normally
  ``config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json``.

Outputs
-------
For each input science image, Step07c writes products in ``config.SISI_ST07_WCS``:

- ``*_wcs_initial.fits``: diagnostic copy of the adopted linear TAN WCS.
- ``*_wcs_master.fits``: protected master-WCS product; created if missing and
  overwritten only with ``--update-master``.
- ``*_wcs.fits``: final operational science image with TAN-SIP WCS.  This is
  the canonical product for Step09 and later stages.

Notes
-----
There is no Step06 orientation dependency in the current T00 workflow.  Step07c
reads directly from Step05 flat-fielded products, so output names preserve the
Step05 stem, e.g. ``Coadd_i_median_078-082_ff_wcs.fits``.

Typical execution
-----------------
PYTHONPATH=. \
SISI_ACTIVE_REDUCTION=config.reductions.sisi_Dolidze25_T00 \
python pipeline/sisi/step07_wcs/step07c_copy_wcs_to_science.py

Useful options
--------------
  --no-sip         write only the linear/TAN WCS
  --update-master  overwrite existing *_wcs_master.fits products
  --wcs-ref FILE   use a different reference WCS FITS file
  --sip-file FILE  use a different SIP calibration JSON file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import Sip, WCS

import config

config.build_sisi_paths()


# -----------------------------------------------------------------------------
# Defaults from the active reduction config
# -----------------------------------------------------------------------------
def resolve_default_wcs_ref() -> Path:
    """
    Resolve the adopted absolute WCS reference.

    The operational reference is config.SISI_WCS_REF_FRAME.  For Dolidze25/T00
    this should point to the final manually adopted quartz WCS, not to the
    older *_wcs_master.fits file that still carries the initial pointing error.

    config.SISI_WCS_REF_MASTER is kept only as a fallback for older reductions.
    """
    tried = []
    for attr in ("SISI_WCS_REF_FINAL", "SISI_WCS_REF_FRAME", "SISI_WCS_REF_MASTER", "SISI_WCS_REF_INITIAL"):
        val = getattr(config, attr, None)
        if val is None:
            continue
        path = Path(val)
        tried.append(f"{attr}={path}")
        if path.exists():
            return path

    raise FileNotFoundError(
        "No valid WCS reference found. Checked:\n  " + "\n  ".join(tried)
    )


WCS_REF_DEFAULT = resolve_default_wcs_ref()
SCI_DIRS_DEFAULT = [Path(config.SISI_ST05_FLATFIELD)]
PATTERNS_DEFAULT = ["*_ff.fits"]
OUT_DIR_DEFAULT = Path(config.SISI_ST07_WCS)


# Include the common WCS/SIP families.  This is intentionally broader than the
# cards normally written by astropy, so stale partial WCS cards cannot survive.
WCS_KEYS_PREFIX = (
    "CD", "PC", "PV", "PS",
    "A_", "B_", "AP_", "BP_",
    "CPDIS", "DP", "D2IM",
)
WCS_KEYS_EXACT = {
    "WCSAXES", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CDELT1", "CDELT2", "CROTA1", "CROTA2",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
}


def strip_wcs_keywords(hdr: fits.Header) -> None:
    """Remove existing WCS/SIP cards before installing the adopted solution."""
    for key in list(hdr.keys()):
        if key in WCS_KEYS_EXACT or key.startswith(WCS_KEYS_PREFIX):
            del hdr[key]


def strip_sip_keywords(hdr: fits.Header) -> None:
    """Remove SIP distortion cards while preserving the linear WCS."""
    for key in list(hdr.keys()):
        if (
            key.startswith(("A_", "B_", "AP_", "BP_"))
            or key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
        ):
            del hdr[key]

    for key in ("CTYPE1", "CTYPE2"):
        if key in hdr and isinstance(hdr[key], str):
            hdr[key] = hdr[key].replace("-SIP", "")


def linear_wcs_from_reference(ref_hdr: fits.Header) -> WCS:
    """
    Build a linear TAN WCS from the adopted reference header.

    The reference may already contain SIP cards.  They are deliberately removed
    here because Step07c attaches the frozen operational SIP model from JSON.
    This prevents stale or duplicate SIP terms from being propagated.
    """
    hdr = ref_hdr.copy()
    strip_sip_keywords(hdr)
    w = WCS(hdr)
    w.sip = None
    for axis in range(2):
        ctype = (w.wcs.ctype[axis] or "").strip()
        if ctype.endswith("-SIP"):
            w.wcs.ctype[axis] = ctype[:-4]
    return w


def print_wcs_diagnostics(w: WCS) -> None:
    """Print the copied reference WCS and its offset from the config target center."""
    crval1, crval2 = w.wcs.crval
    crpix1, crpix2 = w.wcs.crpix
    print(f"Reference CRVAL: RA={crval1:.10f} deg  Dec={crval2:.10f} deg")
    print(f"Reference CRPIX: X={crpix1:.6f}  Y={crpix2:.6f}")
    if hasattr(config, "SISI_RA0_DEG") and hasattr(config, "SISI_DEC0_DEG"):
        dra = (crval1 - float(config.SISI_RA0_DEG)) * 3600.0
        ddec = (crval2 - float(config.SISI_DEC0_DEG)) * 3600.0
        print(
            "Offset from config SISI_RA0/DEC0: "
            f"dRA={dra:+.2f} arcsec, dDec={ddec:+.2f} arcsec"
        )


def _coefficient_dict(cal: dict) -> dict:
    """Return a SIP coefficient dictionary from supported JSON layouts."""
    if "sip_coefficients" in cal:
        return cal["sip_coefficients"]
    return {k: v for k, v in cal.items() if k.startswith(("A_", "B_"))}


def _infer_order(coeffs: dict, prefix: str) -> int:
    order = 0
    for key in coeffs:
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == prefix:
            order = max(order, int(parts[1]) + int(parts[2]))
    return order


def attach_sip_from_json(wcs_obj: WCS, sip_json: Path) -> WCS:
    """Attach frozen SIP coefficients from JSON to a copy of a WCS object."""
    if not sip_json.exists():
        raise FileNotFoundError(f"Missing SIP calibration JSON: {sip_json}")

    with sip_json.open("r") as fh:
        cal = json.load(fh)

    coeffs = _coefficient_dict(cal)
    if not coeffs:
        raise ValueError(f"No SIP coefficients found in {sip_json}")

    a_order = int(cal.get("a_order", cal.get("A_ORDER", _infer_order(coeffs, "A"))))
    b_order = int(cal.get("b_order", cal.get("B_ORDER", _infer_order(coeffs, "B"))))
    order = max(a_order, b_order)

    A = np.zeros((order + 1, order + 1), dtype=float)
    B = np.zeros((order + 1, order + 1), dtype=float)

    for key, val in coeffs.items():
        parts = key.split("_")
        if len(parts) != 3:
            continue
        name, i, j = parts[0], int(parts[1]), int(parts[2])
        if i > order or j > order:
            continue
        if name == "A":
            A[i, j] = float(val)
        elif name == "B":
            B[i, j] = float(val)

    w = wcs_obj.deepcopy()
    w.sip = Sip(A, B, None, None, w.wcs.crpix)

    for axis in range(2):
        ctype = (w.wcs.ctype[axis] or "").strip()
        if ctype and not ctype.endswith("-SIP"):
            w.wcs.ctype[axis] = ctype + "-SIP"

    return w


def default_sip_file() -> Path:
    """Resolve the active SIP calibration path."""
    if hasattr(config, "SISI_SIP_CALIBRATION"):
        return Path(config.SISI_SIP_CALIBRATION)
    return Path(config.PIPELINE_ROOT) / "config" / "calibrations" / "SISI_SIP_SOAR_RUN8_FINAL.json"



def planned_output_files() -> set[Path]:
    """
    Return all files Step07c may write for the active reduction.

    This prevents the adopted WCS reference from being one of the
    products that Step07c is about to overwrite.
    """
    out: set[Path] = set()

    for sci_dir, pattern in zip(SCI_DIRS_DEFAULT, PATTERNS_DEFAULT):
        sci_dir = Path(sci_dir)
        if not sci_dir.exists():
            continue

        for infile in sci_dir.glob(pattern):
            stem = infile.stem
            out.add((OUT_DIR_DEFAULT / f"{stem}_wcs_initial.fits").resolve())
            out.add((OUT_DIR_DEFAULT / f"{stem}_wcs_master.fits").resolve())
            out.add((OUT_DIR_DEFAULT / f"{stem}_wcs.fits").resolve())

    return out


def assert_safe_wcs_reference(wcs_ref: Path) -> None:
    """
    Refuse to use a WCS reference that Step07c may overwrite.

    The adopted WCS reference must be immutable, for example:
      config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits

    It must not be one of the operational output files such as:
      Coadd_i_median_078-082_ff_wcs.fits
      Coadd_r_median_073-077_ff_wcs.fits
    """
    ref = wcs_ref.resolve()
    outputs = planned_output_files()

    if ref in outputs:
        raise RuntimeError(
            "Unsafe Step07c WCS reference.\n"
            "The selected --wcs-ref is one of the files Step07c writes and may overwrite:\n"
            f"  {ref}\n\n"
            "Use an immutable adopted reference instead, for example:\n"
            "  config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits\n"
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the adopted SISI master WCS and SIP calibration to science images."
    )
    parser.add_argument("--wcs-ref", default=str(WCS_REF_DEFAULT),
                        help="Reference FITS file whose adopted absolute linear WCS is copied.")
    parser.add_argument("--sip-file", default=str(default_sip_file()),
                        help="Frozen SIP calibration JSON file.")
    parser.add_argument("--no-sip", action="store_true",
                        help="Do not attach SIP coefficients; write linear/TAN WCS only.")
    parser.add_argument("--update-master", action="store_true",
                        help="Overwrite existing *_wcs_master.fits products.")
    return parser.parse_args()


def write_with_wcs(data: np.ndarray, input_hdr: fits.Header, wcs_hdr: fits.Header,
                   outfile: Path, history: list[str], overwrite: bool = True) -> None:
    hdr = input_hdr.copy()
    strip_wcs_keywords(hdr)
    hdr.update(wcs_hdr)
    for item in history:
        hdr.add_history(item)
    fits.writeto(outfile, data, hdr, overwrite=overwrite)


def main() -> None:
    args = get_args()

    wcs_ref = Path(args.wcs_ref)
    sip_file = Path(args.sip_file)
    apply_sip = not args.no_sip

    if not wcs_ref.exists():
        raise FileNotFoundError(f"Missing WCS reference: {wcs_ref}")

    assert_safe_wcs_reference(wcs_ref)

    with fits.open(wcs_ref, memmap=False) as hdul:
        ref_data = hdul[0].data
        ref_hdr = hdul[0].header

    if ref_data is None:
        raise ValueError(f"Reference WCS file has no primary image data: {wcs_ref}")

    wref = linear_wcs_from_reference(ref_hdr)
    print_wcs_diagnostics(wref)
    linear_hdr = wref.to_header(relax=True)

    sip_hdr = None
    if apply_sip:
        wsip = attach_sip_from_json(wref, sip_file)
        sip_hdr = wsip.to_header(relax=True)

    out_dir = OUT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"WCS reference : {wcs_ref}")
    print(f"Output dir    : {out_dir}")
    print(f"Apply SIP     : {apply_sip}")
    if apply_sip:
        print(f"SIP file      : {sip_file}")

    for sci_dir, pattern in zip(SCI_DIRS_DEFAULT, PATTERNS_DEFAULT):
        sci_dir = Path(sci_dir)
        if not sci_dir.exists():
            print(f"SKIP missing dir: {sci_dir}")
            continue

        files = sorted(sci_dir.glob(pattern))
        if not files:
            print(f"No files matched {pattern!r} in {sci_dir}")
            continue

        for infile in files:
            with fits.open(infile, memmap=False) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header

            if data is None:
                print(f"SKIP no image data: {infile}")
                continue

            if data.shape != ref_data.shape:
                raise ValueError(
                    f"Shape mismatch for {infile.name}: {data.shape} vs {ref_data.shape}. "
                    "The WCS can only be copied directly if science and reference frames "
                    "have identical pixel geometry."
                )

            stem = infile.stem

            initial_out = out_dir / f"{stem}_wcs_initial.fits"
            master_out = out_dir / f"{stem}_wcs_master.fits"
            final_out = out_dir / f"{stem}_wcs.fits"

            history_linear = [f"Linear WCS copied from {wcs_ref.name}"]
            write_with_wcs(data, hdr, linear_hdr, initial_out, history_linear, overwrite=True)
            print(f"Wrote initial : {initial_out}")

            if args.update_master or not master_out.exists():
                write_with_wcs(data, hdr, linear_hdr, master_out, history_linear, overwrite=True)
                print(("Updated" if args.update_master else "Created") + f" master : {master_out}")
            else:
                print(f"Preserved master: {master_out}")

            if apply_sip:
                history_sip = [
                    f"Linear WCS copied from {wcs_ref.name}",
                    f"SIP distortion attached from {sip_file.name}",
                ]
                # Primary operational science product for downstream steps.
                write_with_wcs(data, hdr, sip_hdr, final_out, history_sip, overwrite=True)
                with fits.open(final_out, mode="update", memmap=False) as hdul:
                    hdul[0].header["SIPCAL"] = (True, "Frozen SISI SIP calibration applied")
                    hdul[0].header["SIPFILE"] = (sip_file.name, "SIP calibration JSON file")
                    hdul.flush()
                print(f"Wrote final SIP: {final_out}")

            else:
                write_with_wcs(data, hdr, linear_hdr, final_out, history_linear, overwrite=True)
                print(f"Wrote final TAN: {final_out}")

    print("Done.")


if __name__ == "__main__":
    main()
