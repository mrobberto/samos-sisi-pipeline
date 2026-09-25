#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 11: Calibrate SISI instrumental photometry against SkyMapper.

Inputs
------
products/<target_field>/09_photometry/
    SISI_detected_field_photometry.csv

This catalog is produced by Step09 using the final Step07 TAN-SIP WCS products.

Outputs
-------
products/<target_field>/11_photcal/
    SISI_skymapper_query.csv
    SISI_skymapper_matches.csv
    SISI_skymapper_zero_points.csv
    SISI_detected_field_photometry_skymapper_calibrated.csv

Run
---
PYTHONPATH=. SISI_ACTIVE_REDUCTION=config.reductions.sisi_Dolidze25_T00 \\
python pipeline/sisi/step11_photcal/step11_skymapper_calibrate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

import config

config.build_sisi_paths()


SKYMAPPER_CATALOGS = [
    "II/379/smssdr4",
    "II/379",
]


def active_bands() -> list[str]:
    return list(getattr(config, "SISI_SCIENCE_FRAMES", {}).keys())


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return float(med), float(mad)


def robust_sigma_clip(values, nsig=3.0, maxiter=5):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)

    for _ in range(maxiter):
        if good.sum() < 5:
            break
        med, sig = robust_median_mad(values[good])
        if not np.isfinite(sig) or sig <= 0:
            break
        new_good = good & (np.abs(values - med) <= nsig * sig)
        if new_good.sum() == good.sum():
            good = new_good
            break
        good = new_good

    return good


def find_step09_field_catalog() -> Path:
    p = Path(config.SISI_REDUCED) / "09_photometry" / "SISI_detected_field_photometry.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing Step09 field catalog: {p}")
    return p


def identify_columns(df: pd.DataFrame, bands: list[str]):
    cols = df.columns

    ra_col = first_existing(cols, ["ra_deg", "RA_deg", "ra", "RA"])
    dec_col = first_existing(cols, ["dec_deg", "DEC_deg", "dec", "DEC"])

    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not find RA/Dec columns in {list(cols)}")

    mag_cols = {}
    for b in bands:
        mag_cols[b] = first_existing(
            cols,
            [
                f"mag_inst_{b}",
                f"mag_{b}",
                f"{b}_mag",
                f"inst_mag_{b}",
                f"{b}_inst",
                f"{b}_apmag",
                f"apmag_{b}",
            ],
        )

    missing = [b for b, c in mag_cols.items() if c is None]
    if missing:
        print(f"WARNING: missing instrumental magnitude columns for bands: {missing}")
        print("Available columns:", list(cols))

    return ra_col, dec_col, mag_cols


def query_skymapper(ra_center, dec_center, radius_deg, row_limit=200000, refresh=False):
    cache = Path(config.SISI_SKYMAPPER_CACHE_CSV)

    if cache.exists() and not refresh:
        print(f"Using cached SkyMapper catalog: {cache}")
        return pd.read_csv(cache)

    try:
        from astroquery.vizier import Vizier
    except Exception as exc:
        raise ImportError(
            "astroquery is required for SkyMapper calibration. Install with:\n"
            "  conda install -c conda-forge astroquery\n"
            "or\n"
            "  pip install astroquery"
        ) from exc

    columns = [
        "*",
        "RA_ICRS", "DE_ICRS",
        "RAJ2000", "DEJ2000",
        "rPSF", "e_rPSF",
        "iPSF", "e_iPSF",
        "zPSF", "e_zPSF",
        "gPSF", "e_gPSF",
        "rPetro", "iPetro", "zPetro",
        "ClassStar",
        "flags",
    ]

    v = Vizier(columns=columns, row_limit=row_limit)
    center = SkyCoord(ra_center * u.deg, dec_center * u.deg, frame="icrs")

    last_error = None
    for cat in SKYMAPPER_CATALOGS:
        try:
            print(
                f"Querying VizieR SkyMapper catalog {cat} "
                f"at RA={ra_center:.6f}, Dec={dec_center:.6f}, "
                f"radius={radius_deg:.4f} deg"
            )
            tabs = v.query_region(center, radius=radius_deg * u.deg, catalog=cat)

            if len(tabs) > 0 and len(tabs[0]) > 0:
                sm = tabs[0].to_pandas()
                print(f"SkyMapper rows from {cat}: {len(sm)}")

                cache.parent.mkdir(parents=True, exist_ok=True)
                sm.to_csv(cache, index=False)
                print(f"Cached SkyMapper catalog: {cache}")
                return sm

        except Exception as exc:
            last_error = exc
            print(f"SkyMapper query failed for {cat}: {exc}")

    raise RuntimeError(f"No SkyMapper rows returned. Last error: {last_error}")


def normalize_skymapper_columns(sm: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    sm = sm.copy()

    ra_col = first_existing(
        sm.columns,
        ["sm_ra_deg", "RA_ICRS", "RAICRS", "RAJ2000", "_RAJ2000", "RA"],
    )
    dec_col = first_existing(
        sm.columns,
        ["sm_dec_deg", "DE_ICRS", "DEICRS", "DEJ2000", "_DEJ2000", "DEC"],
    )

    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not identify SkyMapper RA/Dec columns: {list(sm.columns)}")

    sm["sm_ra_deg"] = pd.to_numeric(sm[ra_col], errors="coerce")
    sm["sm_dec_deg"] = pd.to_numeric(sm[dec_col], errors="coerce")

    for b in bands:
        mcol = first_existing(sm.columns, [f"sm_{b}", f"{b}PSF", f"{b}_PSF", f"{b}mag", b])
        ecol = first_existing(sm.columns, [f"sm_e_{b}", f"e_{b}PSF", f"e_{b}_PSF", f"{b}PSFerr", f"e_{b}mag"])

        sm[f"sm_{b}"] = pd.to_numeric(sm[mcol], errors="coerce") if mcol else np.nan
        sm[f"sm_e_{b}"] = pd.to_numeric(sm[ecol], errors="coerce") if ecol else np.nan

    return sm


def match_catalogs(sisi, sm, ra_col, dec_col, max_sep_arcsec=1.0):
    c1 = SkyCoord(
        pd.to_numeric(sisi[ra_col], errors="coerce").to_numpy() * u.deg,
        pd.to_numeric(sisi[dec_col], errors="coerce").to_numpy() * u.deg,
        frame="icrs",
    )
    c2 = SkyCoord(
        pd.to_numeric(sm["sm_ra_deg"], errors="coerce").to_numpy() * u.deg,
        pd.to_numeric(sm["sm_dec_deg"], errors="coerce").to_numpy() * u.deg,
        frame="icrs",
    )

    idx, sep2d, _ = c1.match_to_catalog_sky(c2)
    sep = sep2d.arcsec
    ok = np.isfinite(sep) & (sep <= max_sep_arcsec)

    left = (
        sisi.loc[ok]
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "sisi_index"})
    )
    right = sm.iloc[idx[ok]].copy().reset_index(drop=True)

    matches = pd.concat([left, right.add_prefix("SM_")], axis=1)
    matches["match_sep_arcsec"] = sep[ok]

    for c in [
        "sm_ra_deg", "sm_dec_deg",
        "sm_r", "sm_i", "sm_z", "sm_g",
        "sm_e_r", "sm_e_i", "sm_e_z", "sm_e_g",
    ]:
        pc = "SM_" + c
        if pc in matches:
            matches[c] = matches[pc]

    return matches


def solve_zero_points(matches, mag_cols, bands, nsig=3.0):
    rows = []
    zps = {}

    mag_limits = {
        "r": (18.0, 20.7),
        "i": (18.0, 20.2),
        "z": (18.0, 20.0),
    }

    for b in bands:
        inst_col = mag_cols.get(b)
        sm_col = f"sm_{b}"

        if inst_col is None or sm_col not in matches:
            rows.append(
                dict(
                    band=b,
                    inst_col=inst_col,
                    sm_col=sm_col,
                    n_total=0,
                    n_used=0,
                    zp=np.nan,
                    zp_err=np.nan,
                    scatter=np.nan,
                    nsig_clip=nsig,
                )
            )
            continue

        inst = pd.to_numeric(matches[inst_col], errors="coerce").to_numpy()
        smag = pd.to_numeric(matches[sm_col], errors="coerce").to_numpy()
        delta = smag - inst

        lo, hi = mag_limits.get(b, (5.0, 25.0))

        good = np.isfinite(delta) & np.isfinite(inst) & np.isfinite(smag)
        good &= (smag > lo) & (smag < hi)
        good &= (inst > -50) & (inst < 50)

        err_col = f"sm_e_{b}"
        if err_col in matches:
            e = pd.to_numeric(matches[err_col], errors="coerce").to_numpy()
            good &= (~np.isfinite(e)) | (e < 0.10)

        clipped = np.zeros(len(delta), dtype=bool)
        if good.sum() >= 5:
            clipped_good = robust_sigma_clip(delta[good], nsig=nsig)
            good_idx = np.where(good)[0]
            clipped[good_idx[clipped_good]] = True
        else:
            clipped = good

        zp, scatter = robust_median_mad(delta[clipped])
        n_used = int(clipped.sum())
        zp_err = scatter / np.sqrt(n_used) if n_used > 0 and np.isfinite(scatter) else np.nan

        exptime_s = float(getattr(config, f"SISI_EXPTIME_{b.upper()}", np.nan))
        zp_adus = zp - 2.5 * np.log10(exptime_s) if np.isfinite(exptime_s) and exptime_s > 0 else np.nan

        zps[b] = zp
        matches[f"delta_{b}"] = delta
        matches[f"use_zp_{b}"] = clipped

        print()
        print(f"Band {b}")
        print(f"  instrumental column : {inst_col}")
        print(f"  SkyMapper column    : {sm_col}")
        print(f"  SkyMapper mag range : {lo:.1f} < {b} < {hi:.1f}")
        print(f"  candidates          : {int(good.sum())}")
        print(f"  used after clipping : {n_used}")
        print(f"  ZP                  : {zp:.4f}")
        print(f"  scatter MAD         : {scatter:.4f}")

        rows.append(
            dict(
                band=b,
                inst_col=inst_col,
                sm_col=sm_col,
                n_total=int(good.sum()),
                n_used=n_used,
                zp=zp,
                zp_err=zp_err,
                scatter=scatter,
                nsig_clip=nsig,
                exptime_s=exptime_s,
                zp_adus=zp_adus,
            )
        )

    return pd.DataFrame(rows), matches, zps


def apply_zero_points(df, mag_cols, zps):
    out = df.copy()

    for b, zp in zps.items():
        inst_col = mag_cols.get(b)
        if inst_col is None or not np.isfinite(zp):
            continue
        out[f"mag_{b}_skymapper"] = pd.to_numeric(out[inst_col], errors="coerce") + zp

    if "mag_r_skymapper" in out and "mag_i_skymapper" in out:
        out["r_minus_i_skymapper"] = out["mag_r_skymapper"] - out["mag_i_skymapper"]

    if "mag_i_skymapper" in out and "mag_z_skymapper" in out:
        out["i_minus_z_skymapper"] = out["mag_i_skymapper"] - out["mag_z_skymapper"]

    return out


def amplifier_diagnostics(matches, mag_cols, bands):
    split = float(getattr(config, "SISI_AMP_SPLIT_COL", 528.0))

    xcol = first_existing(
        matches.columns,
        ["x_detect_i", "x_detect_r", "x_i", "x_r", "x"],
    )

    if xcol is None:
        print("WARNING: no x column for amplifier diagnostics.")
        return []

    x = pd.to_numeric(matches[xcol], errors="coerce").to_numpy()
    rows = []

    for b in bands:
        inst_col = mag_cols.get(b)
        sm_col = f"sm_{b}"
        if inst_col is None or sm_col not in matches:
            continue

        inst = pd.to_numeric(matches[inst_col], errors="coerce").to_numpy()
        smag = pd.to_numeric(matches[sm_col], errors="coerce").to_numpy()
        delta = smag - inst

        use_col = f"use_zp_{b}"
        if use_col in matches:
            used = matches[use_col].fillna(False).astype(bool).to_numpy()
        else:
            used = np.isfinite(delta)

        left = used & np.isfinite(x) & (x < split)
        right = used & np.isfinite(x) & (x >= split)

        med_l, mad_l = robust_median_mad(delta[left])
        med_r, mad_r = robust_median_mad(delta[right])
        dz = med_r - med_l if np.isfinite(med_l) and np.isfinite(med_r) else np.nan
        flux_factor = 10 ** (dz / 2.5) if np.isfinite(dz) else np.nan

        rows.append(
            dict(
                band=b,
                xcol=xcol,
                split_x=split,
                n_left=int(left.sum()),
                n_right=int(right.sum()),
                zp_left=med_l,
                zp_right=med_r,
                scatter_left=mad_l,
                scatter_right=mad_r,
                dz_right_minus_left=dz,
                right_flux_factor=flux_factor,
                suggested_scale_multiplier=(1.0 / flux_factor if np.isfinite(flux_factor) and flux_factor > 0 else np.nan),
            )
        )

        print()
        print(f"{b} amplifier diagnostic")
        print(f"  x column             : {xcol}")
        print(f"  left/right N         : {int(left.sum())} / {int(right.sum())}")
        print(f"  left/right ZP        : {med_l:.4f} / {med_r:.4f}")
        print(f"  right-left delta mag : {dz:+.4f}")
        print(f"  right flux factor    : {flux_factor:.4f}")
        print(f"  scale multiplier     : {1.0 / flux_factor if np.isfinite(flux_factor) and flux_factor > 0 else np.nan:.4f}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Calibrate SISI photometry against SkyMapper.")
    parser.add_argument("--input", default=None, help="Input Step09 field photometry CSV.")
    parser.add_argument("--match-radius", type=float, default=1.0, help="Cross-match radius in arcsec.")
    parser.add_argument("--query-pad-arcmin", type=float, default=2.0, help="Pad SkyMapper query radius beyond SISI footprint.")
    parser.add_argument("--row-limit", type=int, default=200000, help="VizieR row limit.")
    parser.add_argument("--refresh-skymapper", action="store_true", help="Refresh cached SkyMapper catalog.")
    parser.add_argument("--nsig", type=float, default=3.0, help="Robust clipping threshold for zero-point solution.")
    args = parser.parse_args()

    bands = active_bands()
    print("Photometric calibration bands:", bands)

    in_csv = Path(args.input) if args.input else find_step09_field_catalog()
    out_dir = Path(config.SISI_REDUCED) / "11_photcal"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SISI photometry: {in_csv}")
    sisi = pd.read_csv(in_csv)

    ra_col, dec_col, mag_cols = identify_columns(sisi, bands)

    ra = pd.to_numeric(sisi[ra_col], errors="coerce").to_numpy()
    dec = pd.to_numeric(sisi[dec_col], errors="coerce").to_numpy()
    ok = np.isfinite(ra) & np.isfinite(dec)

    if ok.sum() == 0:
        raise RuntimeError("No finite RA/Dec values in SISI catalog.")

    ra0 = float(np.nanmedian(ra[ok]))
    dec0 = float(np.nanmedian(dec[ok]))

    dra = (ra[ok] - ra0) * np.cos(np.deg2rad(dec0))
    ddec = dec[ok] - dec0
    radius_deg = float(np.nanmax(np.hypot(dra, ddec)) + args.query_pad_arcmin / 60.0)
    radius_deg = max(radius_deg, 3.0 / 60.0)

    sm_raw = query_skymapper(
        ra0,
        dec0,
        radius_deg,
        row_limit=args.row_limit,
        refresh=args.refresh_skymapper,
    )

    sm = normalize_skymapper_columns(sm_raw, bands)

    sm_out = out_dir / "SISI_skymapper_query.csv"
    sm.to_csv(sm_out, index=False)
    print(f"Wrote SkyMapper working table: {sm_out}")

    matches = match_catalogs(
        sisi,
        sm,
        ra_col,
        dec_col,
        max_sep_arcsec=args.match_radius,
    )

    print(f"Matched SISI↔SkyMapper sources within {args.match_radius:.2f} arcsec: {len(matches)}")

    zp_table, matches, zps = solve_zero_points(matches, mag_cols, bands, nsig=args.nsig)

    amp_rows = amplifier_diagnostics(matches, mag_cols, bands)
    if amp_rows:
        amp_table = pd.DataFrame(amp_rows)
    else:
        amp_table = pd.DataFrame()

    calibrated = apply_zero_points(sisi, mag_cols, zps)

    zp_out = out_dir / "SISI_skymapper_zero_points.csv"
    match_out = out_dir / "SISI_skymapper_matches.csv"
    amp_out = out_dir / "SISI_skymapper_amplifier_diagnostics.csv"
    cal_out = out_dir / "SISI_detected_field_photometry_skymapper_calibrated.csv"

    zp_table.to_csv(zp_out, index=False)
    matches.to_csv(match_out, index=False)
    amp_table.to_csv(amp_out, index=False)
    calibrated.to_csv(cal_out, index=False)

    print()
    print("Wrote:")
    print(" ", sm_out)
    print(" ", match_out)
    print(" ", zp_out)
    print(" ", amp_out)
    print(" ", cal_out)
    print()
    print(zp_table)


if __name__ == "__main__":
    main()