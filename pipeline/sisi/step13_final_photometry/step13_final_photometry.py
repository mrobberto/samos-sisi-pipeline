#!/usr/bin/env python3
"""
SISI Step 13: build final calibrated photometry catalog.

Inputs
------
11_photcal/SISI_skymapper_zero_points.csv
12_psfphot/SISI_psf_photometry_merged.csv

Output
------
13_final_photometry/SISI_final_photometry.csv

Magnitudes are calibrated onto the SkyMapper/AB system using the Step11
zero points. PSF magnitudes are preferred when available; aperture magnitudes
are retained for comparison.

Important convention
--------------------
The final magnitude errors include the external calibration scatter from
the SkyMapper zero-point solution, not only the formal error on the mean:

    magerr_total = sqrt(magerr_phot^2 + zp_scatter^2)

This is intentionally conservative and better represents the absolute
photometric calibration uncertainty.

run:
====
PYTHONPATH=. python drivers/run_sisi_pipeline.py --only 13 --run-qc
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config


def finite_array(x):
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def load_zero_points(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing zero-point table: {path}")

    z = pd.read_csv(path)
    out = {}

    for _, row in z.iterrows():
        band = str(row["band"])
        out[band] = {
            "zp": float(row.get("zp", np.nan)),
            "zp_err": float(row.get("zp_err", np.nan)),
            "scatter": float(row.get("scatter", np.nan)),
            "n_used": int(row.get("n_used", 0)),
            "n_total": int(row.get("n_total", 0)),
        }

    return out


def good_psf_mask(df: pd.DataFrame) -> np.ndarray:
    good = np.ones(len(df), dtype=bool)

    if "psf_flux_fit" in df:
        f = finite_array(df["psf_flux_fit"])
        good &= np.isfinite(f) & (f > 0)
    else:
        good &= False

    if "psf_qfit" in df:
        q = finite_array(df["psf_qfit"])
        good &= np.isfinite(q) & (q < 2.0)

    if "psf_flags" in df:
        flags = finite_array(df["psf_flags"])
        good &= np.isfinite(flags) & (flags == 0)

    if "good_field" in df:
        good &= df["good_field"].astype(bool).to_numpy()

    return good


def main():
    reduced = Path(config.SISI_REDUCED)

    zp_csv = reduced / "11_photcal" / "SISI_skymapper_zero_points.csv"
    psf_csv = reduced / "12_psfphot" / "SISI_psf_photometry_merged.csv"

    out_dir = reduced / "13_final_photometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    zps = load_zero_points(zp_csv)

    if not psf_csv.exists():
        raise FileNotFoundError(f"Missing Step12 merged table: {psf_csv}")

    df = pd.read_csv(psf_csv)

    if "band" not in df:
        raise ValueError("Step12 merged table must contain a 'band' column.")

    bands = list(config.SISI_SCIENCE_FRAMES.keys())

    base_cols = [
        "source_id",
        "ra_deg",
        "dec_deg",
        "detect_band",
        "in_slit_region",
        "wcs_source",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    first_band = bands[0]
    base = df[df["band"].astype(str) == first_band].copy().reset_index(drop=True)
    final = base[base_cols].copy()

    if "x" in base:
        final["x_ref"] = base["x"].to_numpy()
    if "y" in base:
        final["y_ref"] = base["y"].to_numpy()

    for band in bands:
        d = df[df["band"].astype(str) == band].copy().reset_index(drop=True)
        if len(d) == 0:
            print(f"[WARN] no Step12 rows for band {band}; skipping")
            continue

        zp_info = zps.get(band)
        if zp_info is None or not np.isfinite(zp_info["zp"]):
            print(f"[WARN] no valid zero point for band {band}; calibrated mags will be NaN")
            zp = np.nan
            zp_err = np.nan
            zp_scatter = np.nan
        else:
            zp = zp_info["zp"]
            zp_err = zp_info["zp_err"]
            zp_scatter = zp_info["scatter"]

        # Aperture calibrated magnitude.
        if "mag_inst_aper" in d:
            mag_aper_inst = finite_array(d["mag_inst_aper"])
            magerr_aper_phot = (
                finite_array(d["magerr_inst_aper"])
                if "magerr_inst_aper" in d
                else np.full(len(d), np.nan)
            )

            final[f"mag_aper_{band}_AB"] = mag_aper_inst + zp
            final[f"magerr_aper_phot_{band}"] = magerr_aper_phot
            final[f"magerr_aper_{band}"] = np.sqrt(magerr_aper_phot**2 + zp_scatter**2)
            final[f"snr_aper_{band}"] = finite_array(d["snr_aper"]) if "snr_aper" in d else np.nan
            final[f"flux_aper_{band}"] = finite_array(d["flux_aper_bkgsub"]) if "flux_aper_bkgsub" in d else np.nan

        # PSF calibrated magnitude.
        if "psf_flux_fit" in d:
            psf_flux = finite_array(d["psf_flux_fit"])
            mag_psf_inst = np.full(len(d), np.nan)
            ok_flux = np.isfinite(psf_flux) & (psf_flux > 0)
            mag_psf_inst[ok_flux] = -2.5 * np.log10(psf_flux[ok_flux])

            final[f"mag_psf_{band}_AB"] = mag_psf_inst + zp
            final[f"flux_psf_{band}"] = psf_flux

            magerr_psf_phot = np.full(len(d), np.nan)
            if "psf_flux_err" in d:
                psf_fluxerr = finite_array(d["psf_flux_err"])
                ok = ok_flux & np.isfinite(psf_fluxerr) & (psf_fluxerr > 0)
                magerr_psf_phot[ok] = 1.0857362047581294 * psf_fluxerr[ok] / psf_flux[ok]

            final[f"magerr_psf_phot_{band}"] = magerr_psf_phot
            final[f"magerr_psf_{band}"] = np.sqrt(magerr_psf_phot**2 + zp_scatter**2)

            final[f"good_psf_{band}"] = good_psf_mask(d)
            final[f"psf_qfit_{band}"] = finite_array(d["psf_qfit"]) if "psf_qfit" in d else np.nan
            final[f"psf_flags_{band}"] = finite_array(d["psf_flags"]) if "psf_flags" in d else np.nan

        # ----------------------------------------------------------
        # Empirical aperture correction for PSF photometry
        # ----------------------------------------------------------

        psf_col = f"mag_psf_{band}_AB"
        aper_col = f"mag_aper_{band}_AB"
        psf_err_col = f"magerr_psf_{band}"
        aper_err_col = f"magerr_aper_{band}"

        if psf_col in final and aper_col in final:

            delta = (
                final[psf_col].to_numpy(float)
                - final[aper_col].to_numpy(float)
            )

            good_psf_tmp = final[f"good_psf_{band}"].to_numpy(bool)

            use_corr = (
                good_psf_tmp
                & np.isfinite(delta)
            )

            apercorr = np.nanmedian(delta[use_corr])

            final[f"apercorr_{band}"] = apercorr

            final[f"mag_psf_corr_{band}_AB"] = (
                final[psf_col] - apercorr
            )

            print(
                f"{band}: PSF aperture correction = "
                f"{apercorr:.4f} mag"
            )

        # Preferred magnitude: PSF if good, otherwise aperture.
        if (
            f"mag_psf_corr_{band}_AB" in final
            and aper_col in final
        ):

            good_psf = final[f"good_psf_{band}"].to_numpy(dtype=bool)

            final[f"mag_{band}_AB"] = np.where(
                good_psf,
                final[f"mag_psf_corr_{band}_AB"],
                final[aper_col],
            )

            final[f"magerr_{band}_AB"] = np.where(
                good_psf,
                final[psf_err_col],
                final[aper_err_col],
            )

            final[f"mag_source_{band}"] = np.where(
                good_psf,
                "psf_apcorr",
                "aper",
            )

        """
        if psf_col in final and aper_col in final:
            good_psf = final[f"good_psf_{band}"].to_numpy(dtype=bool)
            final[f"mag_{band}_AB"] = np.where(good_psf, final[psf_col], final[aper_col])
            final[f"magerr_{band}_AB"] = np.where(good_psf, final[psf_err_col], final[aper_err_col])
            final[f"mag_source_{band}"] = np.where(good_psf, "psf", "aper")
        elif aper_col in final:
            final[f"mag_{band}_AB"] = final[aper_col]
            final[f"magerr_{band}_AB"] = final[aper_err_col]
            final[f"mag_source_{band}"] = "aper"
        elif psf_col in final:
            final[f"mag_{band}_AB"] = final[psf_col]
            final[f"magerr_{band}_AB"] = final[psf_err_col]
            final[f"mag_source_{band}"] = "psf"
        """

        final[f"zp_{band}"] = zp
        final[f"zp_err_{band}"] = zp_err
        final[f"zp_scatter_{band}"] = zp_scatter

    if "mag_r_AB" in final and "mag_i_AB" in final:
        final["color_r_i_AB"] = final["mag_r_AB"] - final["mag_i_AB"]

    if "magerr_r_AB" in final and "magerr_i_AB" in final:
        final["colorerr_r_i_AB"] = np.sqrt(final["magerr_r_AB"]**2 + final["magerr_i_AB"]**2)

    out_csv = out_dir / "SISI_final_photometry.csv"
    final.to_csv(out_csv, index=False)

    print("=== SISI Step 13: final calibrated photometry ===")
    print(f"Input PSF table : {psf_csv}")
    print(f"Input ZPs       : {zp_csv}")
    print(f"Output          : {out_csv}")
    print(f"N sources       : {len(final)}")
    print("Bands           :", bands)
    print("Error model     : photometric error plus zero-point scatter")
    print("Done.")


if __name__ == "__main__":
    main()
