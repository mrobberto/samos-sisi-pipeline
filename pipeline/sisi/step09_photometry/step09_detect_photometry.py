#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 09: SISI blind-source detection and instrumental aperture photometry.

This step performs field-source photometry on the final SISI science images.
It is intended for broad-band field photometry and CMD construction, not for
forced slit-target photometry.

Production assumptions
----------------------
* Step07 astrometry is frozen and trusted.
* Inputs are the final Step07 science products, i.e. active-band ``*_wcs.fits``
  files carrying the complete TAN-SIP WCS.
* Detection is performed in one reference band, by default ``i`` when available.
* DAOStarFinder provides the initial source positions.
* Each detected source is then recentered with the shared SISI
  ``peak_centered_centroid`` procedure:
      1. start from the DAO position,
      2. search locally for the brightest background-subtracted peak,
      3. compute the centroid in a smaller box around that peak.
* Refined detection-band centroids are converted to RA/Dec.
* Aperture photometry is measured in each active band after projecting those
  RA/Dec coordinates through that band's own WCS.

This avoids assuming that all coadds have exactly the same pixel frame and
ensures that all photometry uses the same centroiding convention.

Outputs
-------
products/<target_field>/09_photometry/
  SISI_detected_all_photometry.csv
  SISI_detected_field_photometry.csv
  SISI_detected_in_slit_photometry.csv
  SISI_detected_field_sources.reg
  SISI_detected_in_slit_sources.reg

The output catalogs include the refined detection-band coordinates, sky
coordinates, aperture photometry in each active band, instrumental colors, slit
mask flags, and DAO/centroiding diagnostics.

No photometric zero point is applied. Magnitudes are instrumental:

    mag_inst = -2.5 log10(net_flux)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from pipeline.sisi.utils.centroiding import peak_centered_centroid

try:
    from photutils.aperture import (
        CircularAnnulus,
        CircularAperture,
        aperture_photometry,
    )
    from photutils.detection import DAOStarFinder
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires photutils. Install in the active environment, e.g.\n"
        "  conda install -c conda-forge photutils\n"
        "or\n"
        "  pip install photutils"
    ) from exc

import config


BAND_RE = re.compile(r"Coadd_([A-Za-z0-9]+)_")
BOX_RE = re.compile(r"box\(([^)]+)\)")


def read_image(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}, got shape={data.shape}")
    data[~np.isfinite(data)] = np.nan
    return data, hdr


def active_science_bands() -> List[str]:
    frames = getattr(config, "SISI_SCIENCE_FRAMES", None)
    if not isinstance(frames, dict) or not frames:
        raise RuntimeError("config.SISI_SCIENCE_FRAMES is missing or empty.")
    return list(frames.keys())


def band_from_name(path: Path) -> str | None:
    m = BAND_RE.search(path.name)
    if m:
        return m.group(1)
    return None


def discover_band_images(wcs_dir: Path, active_bands: Iterable[str]) -> Dict[str, Path]:
    """
    Find final Step07 WCS science images for active science bands only.

    The exact historical filename varied across Step07 revisions, so this
    routine accepts any Coadd_*_wcs*.fits file while rejecting quartz/reference
    frames and stale bands not listed in config.SISI_SCIENCE_FRAMES.
    """
    active = list(active_bands)
    active_set = set(active)

    candidates: List[Path] = []
    for pat in [
        "Coadd_*_wcs.fits",
        "Coadd_*_wcs_master.fits",
        "Coadd_*_wcs_initial.fits",
        "Coadd_*_ff*_wcs.fits",
        "Coadd_*_ff*_wcs_master.fits",
        "Coadd_*_ff*_wcs_initial.fits",
    ]:
        candidates.extend(sorted(wcs_dir.glob(pat)))

    # De-duplicate while preserving order.
    seen = set()
    paths = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if "Quartz" in p.name or "quartz" in p.name:
            continue
        paths.append(p)

    found: Dict[str, Path] = {}
    skipped: List[Tuple[str, Path]] = []

    # Prefer final *_wcs.fits over *_wcs_master/_initial aliases.
    def priority(p: Path) -> tuple[int, str]:
        n = p.name
        if "_wcs.fits" in n and "_master" not in n and "_initial" not in n:
            pri = 0
        elif "_wcs_master.fits" in n:
            pri = 1
        elif "_wcs_initial.fits" in n:
            pri = 2
        else:
            pri = 3
        return (pri, n)

    for p in sorted(paths, key=priority):
        band = band_from_name(p)
        if band is None:
            continue
        if band not in active_set:
            skipped.append((band, p))
            continue
        if band not in found:
            found[band] = p

    missing = [b for b in active if b not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing Step07 WCS images for active bands {missing} in {wcs_dir}. "
            f"Found active bands: {sorted(found)}"
        )

    for band, path in skipped:
        print(f"SKIP {path.name}: band '{band}' is not active in config.SISI_SCIENCE_FRAMES")

    return {b: found[b] for b in active if b in found}


def choose_detection_band(images: Dict[str, Path], preferred: str | None) -> str:
    if preferred:
        if preferred not in images:
            raise ValueError(
                f"Detection band '{preferred}' is not available. "
                f"Available active bands: {sorted(images)}"
            )
        return preferred
    for b in ("i", "r"):
        if b in images:
            return b
    return next(iter(images))


def parse_ds9_boxes(region_path: Path) -> pd.DataFrame:
    rows = []
    if not region_path.exists():
        print(f"[WARN] Slit region file not found; no slit masking: {region_path}")
        return pd.DataFrame(columns=["x", "y", "w", "h", "angle"])

    with open(region_path, "r") as fh:
        for line in fh:
            m = BOX_RE.search(line)
            if not m:
                continue
            parts = [p.strip().replace('"', "") for p in m.group(1).split(",")]
            if len(parts) < 5:
                continue
            try:
                x, y, w, h, ang = map(float, parts[:5])
            except ValueError:
                continue
            rows.append((x, y, w, h, ang))
    return pd.DataFrame(rows, columns=["x", "y", "w", "h", "angle"])


def points_in_rotated_boxes(
    x: np.ndarray,
    y: np.ndarray,
    boxes: pd.DataFrame,
    pad_pix: float = 2.0,
) -> np.ndarray:
    inside = np.zeros(len(x), dtype=bool)
    if boxes.empty:
        return inside

    for _, row in boxes.iterrows():
        xc, yc = float(row.x), float(row.y)
        w = float(row.w) + 2.0 * pad_pix
        h = float(row.h) + 2.0 * pad_pix
        theta = math.radians(float(row.angle))
        dx = x - xc
        dy = y - yc
        xp = dx * math.cos(theta) + dy * math.sin(theta)
        yp = -dx * math.sin(theta) + dy * math.cos(theta)
        inside |= (np.abs(xp) <= 0.5 * w) & (np.abs(yp) <= 0.5 * h)
    return inside


def detect_sources(
    data: np.ndarray,
    fwhm: float,
    threshold_sigma: float,
    sharplo: float,
    sharphi: float,
    roundlo: float,
    roundhi: float,
    edge_margin: float,
    max_sources: int | None = None,
) -> pd.DataFrame:
    finite = np.isfinite(data)
    mean, median, std = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
    print(f"Detection image stats: mean={mean:.3f} median={median:.3f} std={std:.3f}")

    work = data - median
    work[~np.isfinite(work)] = 0.0

    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_sigma * std,
        sharpness_range=(sharplo, sharphi),
        roundness_range=(roundlo, roundhi),
    )
    tbl = finder(work)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError(
            "DAOStarFinder found no sources. Try lowering --threshold-sigma "
            "or adjusting --fwhm."
        )

    df = tbl.to_pandas()
    print("DAOStarFinder columns:", list(df.columns))

    if "xcentroid" in df.columns and "ycentroid" in df.columns:
        df = df.rename(columns={"xcentroid": "x", "ycentroid": "y"})
    elif "x_centroid" in df.columns and "y_centroid" in df.columns:
        df = df.rename(columns={"x_centroid": "x", "y_centroid": "y"})
    elif "x_0" in df.columns and "y_0" in df.columns:
        df = df.rename(columns={"x_0": "x", "y_0": "y"})
    elif "x" in df.columns and "y" in df.columns:
        pass
    else:
        raise RuntimeError(f"Cannot identify source-position columns: {list(df.columns)}")

    ny, nx = data.shape
    ok = (
        np.isfinite(df["x"]) & np.isfinite(df["y"]) &
        (df["x"] >= edge_margin) & (df["x"] <= nx - 1 - edge_margin) &
        (df["y"] >= edge_margin) & (df["y"] <= ny - 1 - edge_margin)
    )
    df = df.loc[ok].copy()

    if max_sources is not None and len(df) > max_sources:
        sort_col = "flux" if "flux" in df.columns else "peak"
        df = df.sort_values(sort_col, ascending=False).head(max_sources).copy()

    df = df.reset_index(drop=True)
    df.insert(0, "source_id", np.arange(1, len(df) + 1, dtype=int))
    print(f"Detected sources after edge cuts: {len(df)}")
    return df


def aperture_measure_band(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ap_radius: float,
    annulus_rin: float,
    annulus_rout: float,
    sigma_clip: SigmaClip,
) -> pd.DataFrame:
    n = len(x)
    out = pd.DataFrame(index=np.arange(n))
    out["x"] = x
    out["y"] = y

    ny, nx = data.shape
    margin = annulus_rout + 1
    inside = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= margin) & (x <= nx - 1 - margin) &
        (y >= margin) & (y <= ny - 1 - margin)
    )
    out["inside_image"] = inside

    for col in ["ap_sum", "bkg_median", "bkg_std", "n_bkg", "flux", "flux_err", "snr", "mag_inst", "mag_err"]:
        out[col] = np.nan

    if not inside.any():
        return out

    pos = np.column_stack([x[inside], y[inside]])
    aper = CircularAperture(pos, r=ap_radius)
    ann = CircularAnnulus(pos, r_in=annulus_rin, r_out=annulus_rout)

    phot = aperture_photometry(data, aper, method="exact")
    ap_sum = np.asarray(phot["aperture_sum"], dtype=float)
    ap_area = float(aper.area)

    bkg_median = np.full(len(pos), np.nan)
    bkg_std = np.full(len(pos), np.nan)
    n_bkg = np.zeros(len(pos), dtype=int)

    masks = ann.to_mask(method="center")
    for i, mask in enumerate(masks):
        cut = mask.multiply(data)
        if cut is None:
            continue
        vals = cut[mask.data > 0]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        vals_clip = sigma_clip(vals)
        vals_good = np.asarray(vals_clip.compressed(), dtype=float)
        n_bkg[i] = len(vals_good)
        if len(vals_good) >= 5:
            bkg_median[i] = np.nanmedian(vals_good)
            bkg_std[i] = np.nanstd(vals_good, ddof=1)

    net = ap_sum - bkg_median * ap_area
    # Include uncertainty in the local sky estimate, using annulus area after clipping.
    flux_err = bkg_std * np.sqrt(ap_area * (1.0 + ap_area / np.maximum(n_bkg, 1)))
    snr = np.divide(net, flux_err, out=np.full_like(net, np.nan), where=flux_err > 0)

    mag = np.full_like(net, np.nan, dtype=float)
    mag_err = np.full_like(net, np.nan, dtype=float)
    good_flux = net > 0
    mag[good_flux] = -2.5 * np.log10(net[good_flux])
    good_err = good_flux & np.isfinite(snr) & (snr > 0)
    mag_err[good_err] = 1.0857362047581296 / snr[good_err]

    idx = np.flatnonzero(inside)
    out.loc[idx, "ap_sum"] = ap_sum
    out.loc[idx, "bkg_median"] = bkg_median
    out.loc[idx, "bkg_std"] = bkg_std
    out.loc[idx, "n_bkg"] = n_bkg
    out.loc[idx, "flux"] = net
    out.loc[idx, "flux_err"] = flux_err
    out.loc[idx, "snr"] = snr
    out.loc[idx, "mag_inst"] = mag
    out.loc[idx, "mag_err"] = mag_err
    return out


def add_band_columns(base: pd.DataFrame, band: str, meas: pd.DataFrame) -> None:
    for col in meas.columns:
        base[f"{col}_{band}"] = meas[col].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SISI blind source detection and aperture photometry.")
    parser.add_argument("--detect-band", default=None, help="Band used for detection; default prefers i, then r.")
    parser.add_argument("--threshold-sigma", type=float, default=5.0, help="DAOStarFinder threshold in sigma.")
    parser.add_argument("--fwhm", type=float, default=5.0, help="DAOStarFinder FWHM in pixels.")
    parser.add_argument("--ap-radius", type=float, default=4.0, help="Aperture radius in pixels.")
    parser.add_argument("--annulus-rin", type=float, default=7.0, help="Sky annulus inner radius in pixels.")
    parser.add_argument("--annulus-rout", type=float, default=12.0, help="Sky annulus outer radius in pixels.")
    parser.add_argument("--sky-sigma", type=float, default=3.0, help="Sigma clipping threshold for local sky annuli.")
    parser.add_argument("--sky-maxiters", type=int, default=5, help="Sigma clipping iterations for local sky annuli.")
    parser.add_argument("--slit-pad", type=float, default=3.0, help="Padding added to slit boxes when flagging excluded sources.")
    parser.add_argument("--edge-margin", type=float, default=15.0, help="Discard detections closer than this many pixels to image edge.")
    parser.add_argument("--max-sources", type=int, default=None, help="Optional cap on number of detected sources, brightest first.")
    parser.add_argument("--sharplo", type=float, default=0.2)
    parser.add_argument("--sharphi", type=float, default=1.5)
    parser.add_argument("--roundlo", type=float, default=-1.0)
    parser.add_argument("--roundhi", type=float, default=1.0)
    args = parser.parse_args()

    wcs_dir = Path(getattr(config, "SISI_ST07_WCS", Path(config.SISI_REDUCED) / "07_wcs"))
    out_dir = Path(getattr(config, "SISI_ST09_PHOTOMETRY", Path(config.SISI_REDUCED) / "09_photometry"))
    out_dir.mkdir(parents=True, exist_ok=True)

    active_bands = active_science_bands()
    print(f"Active science bands from config: {active_bands}")

    images = discover_band_images(wcs_dir, active_bands)
    print("Available active WCS bands:")
    for b, p in images.items():
        print(f"  {b}: {p}")

    det_band = choose_detection_band(images, args.detect_band)
    print(f"Detection band: {det_band}")

    det_data, det_hdr = read_image(images[det_band])
    det_wcs = WCS(det_hdr)

    det = detect_sources(
        det_data,
        fwhm=args.fwhm,
        threshold_sigma=args.threshold_sigma,
        sharplo=args.sharplo,
        sharphi=args.sharphi,
        roundlo=args.roundlo,
        roundhi=args.roundhi,
        edge_margin=args.edge_margin,
        max_sources=args.max_sources,
    )

    #PRESERVE ORIGINAL DAO poitions
    det["x_dao"] = det["x"].to_numpy(float)
    det["y_dao"] = det["y"].to_numpy(float)

    # DAOStarFinder x/y are 0-based pixel coordinates. Convert detections to sky
    # once, then reproject through each band's own WCS for photometry.
    x_refined = det["x_dao"].to_numpy(float).copy()
    y_refined = det["y_dao"].to_numpy(float).copy()
    centroid_shift = np.full(len(det), np.nan)

    for k in range(len(det)):
        xc, yc, shift = peak_centered_centroid(
            det_data,
            x_refined[k],
            y_refined[k],
        )
        centroid_shift[k] = shift
        if np.isfinite(xc) and np.isfinite(yc):
            x_refined[k] = xc
            y_refined[k] = yc

    det["x"] = x_refined
    det["y"] = y_refined
    det["centroid_shift_pix"] = centroid_shift


    ra, dec = det_wcs.all_pix2world(det["x"].to_numpy(), det["y"].to_numpy(), 0)

    cat = det[["source_id", "x", "y"]].copy()
    cat = cat.rename(columns={"x": f"x_detect_{det_band}", "y": f"y_detect_{det_band}"})
    cat["ra_deg"] = ra
    cat["dec_deg"] = dec
    cat["detect_band"] = det_band

    for optional in (
        "x_dao", "y_dao", "centroid_shift_pix",
        "sharpness", "roundness1", "roundness2",
        "npix", "n_pixels", "peak", "flux", "mag", "daofind_mag"
    ):
        if optional in det.columns:
            cat[f"dao_{optional}"] = det[optional].to_numpy()

    # Slit masking is defined in image coordinates for the reference region file.
    # For this T00 product the region file is tied to the final oriented/WCS image
    # frame, so use the detection-band pixel coordinates.
    slit_region = Path(getattr(config, "SISI_PIXEL_REGION", ""))
    boxes = parse_ds9_boxes(slit_region)
    print(f"Slit boxes parsed: {len(boxes)} from {slit_region}")
    cat["in_slit_region"] = points_in_rotated_boxes(
        cat[f"x_detect_{det_band}"].to_numpy(),
        cat[f"y_detect_{det_band}"].to_numpy(),
        boxes,
        pad_pix=args.slit_pad,
    )

    sigma_clip = SigmaClip(sigma=args.sky_sigma, maxiters=args.sky_maxiters)

    for band, path in images.items():
        data, hdr = read_image(path)
        band_wcs = WCS(hdr)
        xb, yb = band_wcs.all_world2pix(cat["ra_deg"].to_numpy(), cat["dec_deg"].to_numpy(), 0)

        meas = aperture_measure_band(
            data,
            np.asarray(xb, dtype=float),
            np.asarray(yb, dtype=float),
            ap_radius=args.ap_radius,
            annulus_rin=args.annulus_rin,
            annulus_rout=args.annulus_rout,
            sigma_clip=sigma_clip,
        )
        add_band_columns(cat, band, meas)
        finite = np.isfinite(meas["mag_inst"]).sum()
        inside = int(meas["inside_image"].sum())
        print(f"Band {band}: inside={inside}/{len(cat)} positive-flux mags={finite}/{len(cat)}")

    if "mag_inst_r" in cat and "mag_inst_i" in cat:
        cat["color_r_i"] = cat["mag_inst_r"] - cat["mag_inst_i"]
    if "mag_inst_i" in cat and "mag_inst_z" in cat:
        cat["color_i_z"] = cat["mag_inst_i"] - cat["mag_inst_z"]
    if "mag_inst_r" in cat and "mag_inst_z" in cat:
        cat["color_r_z"] = cat["mag_inst_r"] - cat["mag_inst_z"]

    all_path = out_dir / "SISI_detected_all_photometry.csv"
    field_path = out_dir / "SISI_detected_field_photometry.csv"
    slit_path = out_dir / "SISI_detected_in_slit_photometry.csv"

    cat.to_csv(all_path, index=False)
    field = cat.loc[~cat["in_slit_region"]].copy()
    slit = cat.loc[cat["in_slit_region"]].copy()
    field.to_csv(field_path, index=False)
    slit.to_csv(slit_path, index=False)

    # DS9 region overlays use detection-band image coordinates.
    xcol = f"x_detect_{det_band}"
    ycol = f"y_detect_{det_band}"
    for name, df, color in [
        ("SISI_detected_field_sources.reg", field, "green"),
        ("SISI_detected_in_slit_sources.reg", slit, "red"),
    ]:
        reg_path = out_dir / name
        with open(reg_path, "w") as fh:
            fh.write("# Region file format: DS9 version 4.1\n")
            fh.write(f'global color={color} width=2 font="helvetica 10 normal"\n')
            fh.write("image\n")
            for _, row in df.iterrows():
                fh.write(f"circle({row[xcol] + 1:.2f},{row[ycol] + 1:.2f},5) # text={{{int(row.source_id)}}}\n")

    print("\nWrote:")
    print(f"  {all_path}       N={len(cat)}")
    print(f"  {field_path}     N={len(field)}  <-- use for CMDs")
    print(f"  {slit_path}      N={len(slit)}   <-- slit-contaminated / diagnostic")
    print(f"  {out_dir / 'SISI_detected_field_sources.reg'}")
    print(f"  {out_dir / 'SISI_detected_in_slit_sources.reg'}")


if __name__ == "__main__":
    main()
