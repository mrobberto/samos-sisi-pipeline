#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISI Step 12: fixed-position PSF photometry at Step09 source positions.

Purpose
-------
Step12 measures PSF-based photometry for the SISI field-source catalog
produced by Step09.  It is intended as a refinement of the Step09 aperture
photometry, not as a new source-detection step.

Current production assumptions
------------------------------
* Step07 astrometry is frozen and trusted.
* Step09 has already detected sources and refined their centroids using the
  shared peak-centered centroid procedure.
* Step12 uses the Step09 source positions as the photometric reference
  positions whenever possible.
* PSF fitting is performed at fixed source positions by default: the PSF fit
  solves for flux only, not for a new centroid.  This makes Step12 consistent
  with the Step09 centroiding convention and avoids noisy nonlinear recentering
  for faint sources.
* A Moffat profile is the default analytic PSF model; Gaussian remains
  available for testing.

Inputs
------
products/<target_field>/09_photometry/
    SISI_detected_field_photometry.csv

products/<target_field>/07_wcs/
    Coadd_<band>_*_ff_wcs.fits

Outputs
-------
products/<target_field>/12_psfphot/
    SISI_psf_photometry_<band>.csv
    SISI_psf_photometry_merged.csv

products/<target_field>/12_psfphot_qc/
    qc_psf_photometry_<band>.reg
    qc_psf_model_<band>.fits
    qc_psf_residual_<band>.fits
    qc_sky_background_<band>.fits
    qc_sky_subtracted_<band>.fits
    qc_sky_mask_<band>.fits

Run
---
PYTHONPATH=. SISI_ACTIVE_REDUCTION=config.reductions.sisi_Dolidze25_T00 \
python pipeline/sisi/step12_psfphot/step12_psf_photometry.py

Useful options
--------------
--bands r i
--no-psf
--psf-model gaussian
--psf-model moffat --moffat-beta 2.5
--force-fwhm 5.5
--free-position
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import QTable, Table
from astropy.wcs import WCS

from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground

try:
    from astropy.modeling.fitting import LevMarLSQFitter
    from photutils.psf import GaussianPSF, MoffatPSF, PSFPhotometry, fit_fwhm
    HAS_PSF = True
except Exception:  # pragma: no cover
    HAS_PSF = False

import config


def build_paths() -> tuple[Path, Path, Path, Path]:
    if hasattr(config, "build_sisi_paths"):
        config.build_sisi_paths()

    reduced = Path(config.SISI_REDUCED)
    wcs_dir = Path(getattr(config, "SISI_ST07_WCS", reduced / "07_wcs"))
    out_dir = Path(getattr(config, "SISI_ST12_PSFPHOT", reduced / "12_psfphot"))
    qc_dir = Path(getattr(config, "SISI_ST12_PSFPHOT_QC", reduced / "12_psfphot_qc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    return reduced, wcs_dir, out_dir, qc_dir


def finite_array(x) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def read_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}, got shape {data.shape}")
    data[~np.isfinite(data)] = np.nan
    return data, hdr


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def infer_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(
        f"Could not identify {kind} column. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def load_catalog(path: Path) -> pd.DataFrame:
    """Load the Step09 catalog and normalize RA, Dec, and source_id columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing catalog: {path}")

    df = pd.read_csv(path, comment="#")
    bad_header = all(str(c).startswith("Unnamed") or str(c).isdigit() for c in df.columns)

    if bad_header:
        df = pd.read_csv(path, comment="#", header=None)
        if df.shape[1] >= 3:
            df = df.rename(columns={0: "source_id", 1: "ra_deg", 2: "dec_deg"})
        elif df.shape[1] == 2:
            df = df.rename(columns={0: "ra_deg", 1: "dec_deg"})
        else:
            raise ValueError(f"Catalog {path} has too few columns")

    ra_col = infer_column(df, ["ra_deg", "ra", "RA", "RA_deg", "radeg", "RAJ2000"], "RA")
    dec_col = infer_column(df, ["dec_deg", "dec", "DEC", "DEC_deg", "decdeg", "DEJ2000"], "Dec")

    id_col = None
    for cand in ["source_id", "id", "object_id", "name", "slit_id", "slit"]:
        try:
            id_col = infer_column(df, [cand], "ID")
            break
        except KeyError:
            pass

    out = df.copy()
    out["ra_deg"] = pd.to_numeric(out[ra_col], errors="coerce")
    out["dec_deg"] = pd.to_numeric(out[dec_col], errors="coerce")

    if id_col is not None:
        out["source_id"] = out[id_col].astype(str)
    else:
        out["source_id"] = [f"SRC{i:04d}" for i in range(len(out))]

    good = np.isfinite(out["ra_deg"]) & np.isfinite(out["dec_deg"])
    out = out.loc[good].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No finite RA/Dec rows found in {path}")

    return out


def active_bands() -> list[str]:
    frames = getattr(config, "SISI_SCIENCE_FRAMES", None)
    if isinstance(frames, dict) and frames:
        return list(frames.keys())
    return ["r", "i"]


def find_step09_catalog(reduced: Path) -> Path:
    phot_dir = reduced / "09_photometry"
    candidates = [
        phot_dir / "SISI_detected_field_photometry.csv",
        phot_dir / "SISI_detected_all_photometry.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing Step09 photometry catalog. Tried:\n" +
        "\n".join(str(p) for p in candidates)
    )


def find_band_image(wcs_dir: Path, band: str) -> Path:
    """Find the final Step07 WCS image for one band."""
    patterns = [
        f"Coadd_{band}_*_ff_wcs.fits",
        f"Coadd_{band}_*_wcs.fits",
        f"Coadd_{band}_*_ff*_wcs.fits",
        f"Coadd_{band}_*_ff_wcs_master.fits",
        f"Coadd_{band}_*_ff_wcs_initial.fits",
        # Historical fallbacks only.
        f"Coadd_{band}_*_ff_orient_wcs.fits",
        f"Coadd_{band}_*_ff_orient_wcs_master.fits",
        f"Coadd_{band}_*_ff_orient_wcs_initial.fits",
    ]

    def priority(p: Path) -> tuple[int, str]:
        n = p.name
        if n.endswith("_ff_wcs.fits"):
            pri = 0
        elif n.endswith("_wcs.fits") and "_master" not in n and "_initial" not in n:
            pri = 1
        elif "_master" in n:
            pri = 2
        elif "_initial" in n:
            pri = 3
        else:
            pri = 4
        return pri, n

    hits: list[Path] = []
    for pat in patterns:
        hits.extend(wcs_dir.glob(pat))
    hits = sorted({p for p in hits if "Quartz" not in p.name and "quartz" not in p.name}, key=priority)

    if not hits:
        raise FileNotFoundError(
            f"No Step07 WCS image found for band {band}. Tried patterns in {wcs_dir}:\n" +
            "\n".join(patterns)
        )
    return hits[0]


def choose_positions(cat: pd.DataFrame, band: str, w: WCS) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Choose input positions for Step12.

    Prefer Step09 measured detector coordinates.  If the catalog was detected
    in band i, columns are typically x_detect_i/y_detect_i.  These are already
    peak-centered positions and should be used directly for the detection band.
    For other bands, prefer the Step09 per-band projected coordinates x_<band>,
    y_<band>, falling back to WCS projection from RA/Dec.
    """
    pairs = [
        (f"x_detect_{band}", f"y_detect_{band}", f"step09_detect_xy_{band}"),
        (f"x_{band}", f"y_{band}", f"step09_band_xy_{band}"),
        ("x", "y", "generic_xy"),
    ]

    for xc, yc, label in pairs:
        if xc in cat.columns and yc in cat.columns:
            x = finite_array(cat[xc])
            y = finite_array(cat[yc])
            if np.isfinite(x).sum() > 0 and np.isfinite(y).sum() > 0:
                return x, y, label

    x, y = w.all_world2pix(cat["ra_deg"].to_numpy(float), cat["dec_deg"].to_numpy(float), 0)
    return np.asarray(x, float), np.asarray(y, float), "wcs_radec"


def make_smooth_sky_background(
    data: np.ndarray,
    box_size: int = 192,
    filter_size: int = 3,
    sigma: float = 3.0,
    edge_mask: int = 40,
    bad_percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(data)
    good_vals = data[finite]
    lo, hi = np.nanpercentile(good_vals, [0.5, bad_percentile])

    mask = ~finite | (data < lo) | (data > hi)
    if edge_mask > 0:
        mask[:edge_mask, :] = True
        mask[-edge_mask:, :] = True
        mask[:, :edge_mask] = True
        mask[:, -edge_mask:] = True

    sigma_clip = SigmaClip(sigma=sigma, maxiters=5)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        data,
        box_size=(box_size, box_size),
        filter_size=(filter_size, filter_size),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
        mask=mask,
        exclude_percentile=60.0,
    )

    sky = np.asarray(bkg.background, dtype=float)
    return sky, data - sky, mask

def make_split_smooth_sky_background(
    data: np.ndarray,
    split_col: int,
    box_size: int = 192,
    filter_size: int = 3,
    sigma: float = 3.0,
    edge_mask: int = 40,
    bad_percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sky = np.full_like(data, np.nan, dtype=float)
    mask = np.zeros_like(data, dtype=bool)

    for x0, x1 in [(0, split_col), (split_col, data.shape[1])]:
        sub = data[:, x0:x1]

        sub_sky, sub_bkgsub, sub_mask = make_smooth_sky_background(
            sub,
            box_size=box_size,
            filter_size=filter_size,
            sigma=sigma,
            edge_mask=edge_mask,
            bad_percentile=bad_percentile,
        )

        sky[:, x0:x1] = sub_sky
        mask[:, x0:x1] = sub_mask

    return sky, data - sky, mask


def aperture_fluxes(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    aperture_radius: float,
    annulus_rin: float,
    annulus_rout: float,
) -> pd.DataFrame:
    positions = np.column_stack([x, y])
    aper = CircularAperture(positions, r=aperture_radius)
    ann = CircularAnnulus(positions, r_in=annulus_rin, r_out=annulus_rout)

    phot = aperture_photometry(data, aper, method="exact")
    aper_sum = np.asarray(phot["aperture_sum"], dtype=float)
    aper_area = float(aper.area)

    bkg_median = np.full(len(x), np.nan)
    bkg_std = np.full(len(x), np.nan)
    n_ann = np.zeros(len(x), dtype=int)

    masks = ann.to_mask(method="center")
    for i, mask in enumerate(masks):
        ann_data = mask.multiply(data)
        if ann_data is None:
            continue
        vals = ann_data[mask.data > 0]
        vals = vals[np.isfinite(vals)]
        n_ann[i] = len(vals)
        if len(vals) >= 5:
            _, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
            bkg_median[i] = med
            bkg_std[i] = std

    flux = aper_sum - bkg_median * aper_area
    flux_err = bkg_std * np.sqrt(aper_area * (1.0 + aper_area / np.maximum(n_ann, 1)))
    snr = np.divide(flux, flux_err, out=np.full_like(flux, np.nan), where=flux_err > 0)

    mag = np.full(len(flux), np.nan)
    magerr = np.full(len(flux), np.nan)
    pos = flux > 0
    mag[pos] = -2.5 * np.log10(flux[pos])
    good_err = pos & np.isfinite(snr) & (snr > 0)
    magerr[good_err] = 1.0857362047581296 / snr[good_err]

    return pd.DataFrame({
        "aperture_sum": aper_sum,
        "bkg_median_annulus": bkg_median,
        "bkg_std_annulus": bkg_std,
        "n_annulus_pix": n_ann,
        "flux_aper_bkgsub": flux,
        "fluxerr_aper": flux_err,
        "snr_aper": snr,
        "mag_inst_aper": mag,
        "magerr_inst_aper": magerr,
    })


def estimate_fwhm(data_bkgsub: np.ndarray, x: np.ndarray, y: np.ndarray, default: float) -> float:
    if not HAS_PSF or len(x) == 0:
        return default

    try:
        xypos = list(zip(x, y))
        vals = fit_fwhm(data_bkgsub, xypos=xypos, fit_shape=(11, 11), fwhm=default)
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0.5) & (vals < 20.0)]
        if len(vals) >= 3:
            return float(np.nanmedian(vals))
    except Exception as exc:
        print(f"  [WARN] FWHM estimate failed; using {default:.2f}: {exc}")

    return default


def build_psf_model(fwhm: float, model: str = "moffat", moffat_beta: float = 2.5, free_position: bool = False):
    model = model.lower()
    if model == "gaussian":
        psf = GaussianPSF(
            flux=1.0,
            x_0=0.0,
            y_0=0.0,
            x_fwhm=fwhm,
            y_fwhm=fwhm,
        )
    elif model == "moffat":
        alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / moffat_beta) - 1.0))
        psf = MoffatPSF(
            flux=1.0,
            x_0=0.0,
            y_0=0.0,
            alpha=alpha,
            beta=moffat_beta,
        )
    else:
        raise ValueError(f"Unknown PSF model: {model}")

    # Production default: fixed-position PSF photometry.  Step09 owns centroids.
    psf.x_0.fixed = not free_position
    psf.y_0.fixed = not free_position
    return psf


def psf_fluxes(
    data_bkgsub: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fwhm: float,
    fit_shape: int,
    psf_model_name: str,
    moffat_beta: float,
    free_position: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    if not HAS_PSF:
        raise RuntimeError("photutils PSF tools are not available in this environment")

    init = QTable()
    init["x"] = x
    init["y"] = y

    psf_model = build_psf_model(
        fwhm=fwhm,
        model=psf_model_name,
        moffat_beta=moffat_beta,
        free_position=free_position,
    )

    psfphot = PSFPhotometry(
        psf_model=psf_model,
        fit_shape=(fit_shape, fit_shape),
        fitter=LevMarLSQFitter(),
        aperture_radius=max(3.0, fwhm),
    )

    phot = psfphot(data_bkgsub, init_params=init)
    table = Table(phot).to_pandas()

    model_image = None
    residual_image = None
    try:
        model_image = psfphot.make_model_image(data_bkgsub.shape)
        residual_image = data_bkgsub - model_image
    except Exception as exc:
        print(f"  [WARN] Could not build PSF model/residual image: {exc}")

    out = pd.DataFrame(index=np.arange(len(x)))
    for col in table.columns:
        out[f"psf_{col}"] = table[col]
    return out, model_image, residual_image


def write_ds9_region(path: Path, df: pd.DataFrame, color: str = "green") -> None:
    with open(path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f'global color={color} width=2 font="helvetica 10 normal"\n')
        f.write("image\n")
        for _, row in df.iterrows():
            sid = str(row.get("source_id", ""))
            x = float(row["x"])
            y = float(row["y"])
            f.write(f"circle({x + 1:.2f},{y + 1:.2f},5) # text={{{sid}}}\n")


def process_band(
    band: str,
    image_path: Path,
    cat: pd.DataFrame,
    out_dir: Path,
    qc_dir: Path,
    aperture_radius: float,
    annulus_rin: float,
    annulus_rout: float,
    default_fwhm: float,
    force_fwhm: float | None,
    do_psf: bool,
    psf_fit_shape: int,
    psf_model_name: str,
    moffat_beta: float,
    free_position: bool,
    sky_box_size: int,
) -> pd.DataFrame:
    data, hdr = read_image(image_path)
    w = WCS(hdr)
    ny, nx = data.shape

    x, y, position_source = choose_positions(cat, band, w)

    margin = max(aperture_radius, annulus_rout, psf_fit_shape / 2 if do_psf else 0) + 2
    inside = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= margin) & (x <= nx - 1 - margin) &
        (y >= margin) & (y <= ny - 1 - margin)
    )

    edge_margin = 40.0
    good_field = (
        inside &
        (x >= edge_margin) & (x <= nx - 1 - edge_margin) &
        (y >= edge_margin) & (y <= ny - 1 - edge_margin)
    )

    out = cat.copy()
    out["band"] = band
    out["image"] = image_path.name
    out["x"] = x
    out["y"] = y
    out["inside_image"] = inside
    out["good_field"] = good_field
    out["position_source"] = position_source

    _, med, std = sigma_clipped_stats(data[np.isfinite(data)], sigma=3.0, maxiters=5)

    try:
        split_col = int(getattr(config, "SISI_AMP_SPLIT_COL", data.shape[1] // 2))

        sky_image, data_bkgsub, sky_mask = make_split_smooth_sky_background(
            data,
            split_col=split_col,
            box_size=sky_box_size,
            filter_size=3,
            sigma=3.0,
            edge_mask=40,
            bad_percentile=99.0,
        )

        fits.writeto(qc_dir / f"qc_sky_background_{band}.fits", sky_image.astype(np.float32), hdr, overwrite=True)
        fits.writeto(qc_dir / f"qc_sky_subtracted_{band}.fits", data_bkgsub.astype(np.float32), hdr, overwrite=True)
        fits.writeto(qc_dir / f"qc_sky_mask_{band}.fits", sky_mask.astype(np.uint8), hdr, overwrite=True)
        out["background_model"] = "Background2D_split_amp"
        out["background_box_size"] = sky_box_size
        out["background_filter_size"] = 3
    except Exception as exc:
        print(f"  [WARN] Background2D failed for band {band}: {exc}")
        print("  [WARN] Falling back to global median background.")
        data_bkgsub = data - med
        out["background_model"] = "global_median"
        out["background_box_size"] = np.nan
        out["background_filter_size"] = np.nan

    out["global_bkg_median"] = med
    out["global_bkg_std"] = std

    print(f"\nBand {band}: {image_path.name}")
    print(f"  image shape: {nx} x {ny}")
    print(f"  position source: {position_source}")
    print(f"  catalog rows: {len(cat)} ; inside={int(inside.sum())} ; good_field={int(good_field.sum())}")
    print(f"  global background median={med:.4g}, std={std:.4g}")

    phot = pd.DataFrame(index=out.index)
    if not good_field.any():
        out["fwhm_est_pix"] = np.nan
    else:
        ap = aperture_fluxes(data, x[good_field], y[good_field], aperture_radius, annulus_rin, annulus_rout)
        for col in ap.columns:
            phot[col] = np.nan
            phot.loc[good_field, col] = ap[col].to_numpy()

        if force_fwhm is not None:
            fwhm = float(force_fwhm)
        elif do_psf:
            fwhm = estimate_fwhm(data_bkgsub, x[good_field], y[good_field], default_fwhm)
        else:
            fwhm = float(default_fwhm)

        out["fwhm_est_pix"] = fwhm
        print(f"  FWHM estimate for PSF/QC: {fwhm:.3f} pix")

        if do_psf:
            try:
                psf, model_image, residual_image = psf_fluxes(
                    data_bkgsub,
                    x[good_field],
                    y[good_field],
                    fwhm,
                    psf_fit_shape,
                    psf_model_name,
                    moffat_beta,
                    free_position=free_position,
                )

                for col in psf.columns:
                    phot[col] = np.nan
                    phot.loc[good_field, col] = psf[col].to_numpy()

                # Flux column naming differs slightly between photutils versions.
                flux_col = "psf_flux_fit" if "psf_flux_fit" in phot.columns else None
                if flux_col is None:
                    for cand in ["psf_flux", "psf_flux_0"]:
                        if cand in phot.columns:
                            flux_col = cand
                            break

                if flux_col is not None:
                    flux = finite_array(phot[flux_col])
                    phot["mag_psf_inst"] = np.nan
                    ok_flux = np.isfinite(flux) & (flux > 0)
                    phot.loc[ok_flux, "mag_psf_inst"] = -2.5 * np.log10(flux[ok_flux])
                else:
                    phot["mag_psf_inst"] = np.nan

                if free_position and "psf_x_fit" in phot.columns and "psf_y_fit" in phot.columns:
                    phot["psf_dx"] = np.nan
                    phot["psf_dy"] = np.nan
                    phot.loc[good_field, "psf_dx"] = phot.loc[good_field, "psf_x_fit"].to_numpy(float) - x[good_field]
                    phot.loc[good_field, "psf_dy"] = phot.loc[good_field, "psf_y_fit"].to_numpy(float) - y[good_field]
                    phot["psf_shift_pix"] = np.hypot(phot["psf_dx"], phot["psf_dy"])
                else:
                    phot["psf_dx"] = 0.0
                    phot["psf_dy"] = 0.0
                    phot["psf_shift_pix"] = 0.0

                flags = phot["psf_flags"] if "psf_flags" in phot.columns else pd.Series(0, index=phot.index)
                flux_for_good = finite_array(phot[flux_col]) if flux_col is not None else np.full(len(phot), np.nan)
                phot["good_psf_fit"] = (
                    np.isfinite(flux_for_good) &
                    (flux_for_good > 0) &
                    np.isfinite(phot["psf_shift_pix"]) &
                    (phot["psf_shift_pix"] < (1.0 if free_position else 0.5)) &
                    (pd.to_numeric(flags, errors="coerce").fillna(999).to_numpy() == 0)
                )

                print(f"  PSF photometry: OK ; good fits={int(phot['good_psf_fit'].sum())}")

                if model_image is not None:
                    fits.writeto(qc_dir / f"qc_psf_model_{band}.fits", model_image.astype(np.float32), hdr, overwrite=True)
                if residual_image is not None:
                    fits.writeto(qc_dir / f"qc_psf_residual_{band}.fits", residual_image.astype(np.float32), hdr, overwrite=True)

            except Exception as exc:
                print(f"  [WARN] PSF photometry failed for band {band}: {exc}")

    out = pd.concat([out, phot], axis=1)
    out["psf_model"] = psf_model_name
    out["psf_fixed_position"] = not free_position
    out["psf_beta"] = moffat_beta if psf_model_name == "moffat" else np.nan
    out["aperture_radius"] = aperture_radius
    out["annulus_rin"] = annulus_rin
    out["annulus_rout"] = annulus_rout

    out_csv = out_dir / f"SISI_psf_photometry_{band}.csv"
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    reg_csv = qc_dir / f"qc_psf_photometry_{band}.reg"
    write_ds9_region(reg_csv, out.loc[good_field].copy())
    print(f"  wrote {reg_csv}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SISI fixed-position PSF photometry from Step09 sources.")
    parser.add_argument("--bands", nargs="*", default=None, help="Bands to process; default: config.SISI_SCIENCE_FRAMES keys")
    parser.add_argument("--input", default=None, help="Input Step09 photometry catalog; default: field catalog")
    parser.add_argument("--aperture-radius", type=float, default=5.0)
    parser.add_argument("--annulus-rin", type=float, default=8.0)
    parser.add_argument("--annulus-rout", type=float, default=12.0)
    parser.add_argument("--default-fwhm", type=float, default=5.5)
    parser.add_argument("--force-fwhm", type=float, default=None)
    parser.add_argument("--no-psf", action="store_true", help="Disable PSF photometry and run aperture diagnostics only")
    parser.add_argument("--psf-fit-shape", type=int, default=21)
    parser.add_argument("--psf-model", choices=["gaussian", "moffat"], default="moffat")
    parser.add_argument("--moffat-beta", type=float, default=2.5)
    parser.add_argument("--free-position", action="store_true", help="Allow PSF fit to recenter sources. Default is fixed-position photometry.")
    parser.add_argument("--sky-box-size", type=int, default=192)
    args = parser.parse_args()

    if args.psf_fit_shape % 2 == 0:
        raise ValueError("--psf-fit-shape must be odd")

    reduced, wcs_dir, out_dir, qc_dir = build_paths()

    bands = args.bands if args.bands is not None else active_bands()
    cat_path = Path(args.input) if args.input else find_step09_catalog(reduced)
    cat = load_catalog(cat_path)

    print("=== SISI Step 12: fixed-position PSF photometry ===")
    print(f"Catalog: {cat_path}")
    print(f"WCS dir: {wcs_dir}")
    print(f"Output : {out_dir}")
    print(f"QC     : {qc_dir}")
    print(f"Bands  : {bands}")
    print(f"Aperture r={args.aperture_radius}, annulus=({args.annulus_rin},{args.annulus_rout}) pix")
    print(f"PSF model: {args.psf_model}; fixed_position={not args.free_position}")
    if args.psf_model == "moffat":
        print(f"Moffat beta: {args.moffat_beta:.2f}")

    all_rows = []
    for band in bands:
        image_path = find_band_image(wcs_dir, band)
        print(f"Using {band} image: {image_path}")

        rows = process_band(
            band=band,
            image_path=image_path,
            cat=cat,
            out_dir=out_dir,
            qc_dir=qc_dir,
            aperture_radius=args.aperture_radius,
            annulus_rin=args.annulus_rin,
            annulus_rout=args.annulus_rout,
            default_fwhm=args.default_fwhm,
            force_fwhm=args.force_fwhm,
            do_psf=not args.no_psf,
            psf_fit_shape=args.psf_fit_shape,
            psf_model_name=args.psf_model,
            moffat_beta=args.moffat_beta,
            free_position=args.free_position,
            sky_box_size=args.sky_box_size,
        )
        all_rows.append(rows)

    merged = pd.concat(all_rows, ignore_index=True)
    merged_csv = out_dir / "SISI_psf_photometry_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\nWrote merged table: {merged_csv}")
    print("=== Step 12 complete ===")


if __name__ == "__main__":
    main()
