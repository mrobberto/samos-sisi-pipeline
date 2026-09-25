#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISI Step14: merge T00 and T01 final calibrated catalogs.

Purpose
-------
Build a final Dolidze25 SISI photometric catalog on the SkyMapper/AB system,
using the independently calibrated Step13 catalogs for T00 and T01.

The script also identifies sources observed in both fields.  For overlap
sources where one observation falls on/near a SAMOS slit/bar and the other
does not, it keeps the two measurements separated so that the flux loss caused
by the bar can be estimated.

Inputs
------
products/Dolidze25_T00/13_final_photometry/SISI_final_photometry*.csv
products/Dolidze25_T01/13_final_photometry/SISI_final_photometry*.csv

Outputs
-------
products/Dolidze25_merged_photometry/
    SISI_merged_photometry.csv
    SISI_overlap_matches_all.csv
    SISI_overlap_bar_loss.csv

Convention
----------
T01 is placed on the T00/SkyMapper mean system using empirical overlap
corrections derived from clean common stars. The correction is solved robustly:
a preliminary clean sample is built from slit/bar and PSF-quality flags, an
initial T01-T00 offset is measured, catastrophic overlap outliers are rejected,
and the correction is recomputed from the clipped sample.

No z overlap correction is applied by default because T00 z is not part of
the production photometry. z remains on the direct SkyMapper Step11 calibration.

Run
---
PYTHONPATH=. python pipeline/sisi/step14_merge/step14_merge_T00_T01.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u


DEFAULT_REL_CORR = {
    "r": 0.0,
    "i": 0.0,
    "z": 0.0,
}


def finite_array(x):
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return float(med), float(mad), int(len(x))


def derive_one_band_correction(raw, base_good, *, max_resid=0.50, nsigma=3.0,
                               min_n=8, n_iter=5):
    """Return T01->T00 correction and robust inlier mask for one band.

    raw is T01 - T00 before correction. The returned correction is added to
    T01, so the corrected residual is raw + corr. The hard residual cut prevents
    catastrophic 1--3 mag overlap failures from defining the inter-field zero
    point.
    """
    raw = np.asarray(raw, dtype=float)
    base_good = np.asarray(base_good, dtype=bool)
    finite = base_good & np.isfinite(raw)

    if finite.sum() < min_n:
        return 0.0, np.zeros(len(raw), dtype=bool), np.nan, np.nan, int(finite.sum())

    keep = finite.copy()
    corr = 0.0

    for _ in range(n_iter):
        med, mad, n = robust_median_mad(raw[keep])
        if n < min_n or not np.isfinite(med):
            break

        corr = -med
        resid = raw + corr

        lim = float(max_resid)
        if np.isfinite(mad) and mad > 0:
            lim = min(lim, float(nsigma) * mad)

        new_keep = finite & np.isfinite(resid) & (np.abs(resid) <= lim)
        if new_keep.sum() < min_n:
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    med, mad, n = robust_median_mad(raw[keep])
    if n >= min_n and np.isfinite(med):
        corr = -med

    return float(corr), keep, float(med), float(mad), int(n)

def find_catalog(field_dir: Path) -> Path:
    field = field_dir.name.split("_")[-1]

    candidates = [
        field_dir / "13_final_photometry" / f"SISI_final_photometry_{field}.csv",
        field_dir / "13_final_photometry" / "SISI_final_photometry.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(
        (field_dir / "13_final_photometry").glob("SISI_final_photometry*.csv")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No Step13 final photometry catalog found in "
        f"{field_dir / '13_final_photometry'}"
    )

def match_catalogs(cat0: pd.DataFrame, cat1: pd.DataFrame, radius_arcsec: float = 2.0) -> pd.DataFrame:
    c0 = SkyCoord(
        cat0["ra_deg"].to_numpy() * u.deg,
        cat0["dec_deg"].to_numpy() * u.deg,
    )
    c1 = SkyCoord(
        cat1["ra_deg"].to_numpy() * u.deg,
        cat1["dec_deg"].to_numpy() * u.deg,
    )

    idx, sep2d, _ = c0.match_to_catalog_sky(c1)
    ok = np.isfinite(sep2d.arcsec) & (sep2d.arcsec <= radius_arcsec)

    m0 = cat0.loc[ok].copy().reset_index(drop=True)
    m1 = cat1.iloc[idx[ok]].copy().reset_index(drop=True)

    out = pd.concat([m0.add_suffix("_T00"), m1.add_suffix("_T01")], axis=1)
    out["match_sep_arcsec"] = sep2d.arcsec[ok]

    return out

def apply_overlap_quality_and_corrections(matches: pd.DataFrame, corr: dict[str, float],
                                          auto_bands: set[str], *,
                                          max_resid=0.50, nsigma=3.0, min_n=8):
    """Finalize overlap-quality flags and derive requested T01 corrections."""
    base_good = ~matches["bad_overlap"].fillna(False).astype(bool).to_numpy()
    final_good = base_good.copy()
    matches["overlap_base_good"] = base_good

    print("\nDeriving/checking T01 corrections from clean PSF overlap:")

    for b in ["r", "i"]:
        col0 = f"mag_{b}_AB_T00"
        col1 = f"mag_{b}_AB_T01"
        if col0 not in matches.columns or col1 not in matches.columns:
            matches[f"use_overlap_{b}"] = False
            final_good &= False
            continue

        raw = finite_array(matches[col1]) - finite_array(matches[col0])

        if b in auto_bands:
            c, keep, med, mad, n = derive_one_band_correction(
                raw, base_good, max_resid=max_resid, nsigma=nsigma, min_n=min_n
            )
            corr[b] = c
            print(
                f"  {b}: clipped raw median T01-T00={med:.4f}, "
                f"sigmaMAD={mad:.4f}, N={n} -> corr={corr[b]:+.4f}"
            )
        else:
            resid = raw + corr.get(b, 0.0)
            keep = base_good & np.isfinite(resid) & (np.abs(resid) <= max_resid)
            med, mad, n = robust_median_mad(resid[keep])
            print(
                f"  {b}: user corr={corr[b]:+.4f}; clipped residual "
                f"median={med:.4f}, sigmaMAD={mad:.4f}, N={n}"
            )

        matches[f"use_overlap_{b}"] = keep
        final_good &= keep

    matches["use_overlap_photcal"] = final_good

    for b in ["r", "i"]:
        col0 = f"mag_{b}_AB_T00"
        col1c = f"mag_{b}_AB_T01_corr"
        if col0 in matches.columns and col1c in matches.columns:
            resid = finite_array(matches[col1c]) - finite_array(matches[col0])
            good = matches[f"use_overlap_{b}"].fillna(False).astype(bool).to_numpy()
            med, mad, n = robust_median_mad(resid[good])
            print(
                f"  {b}: final corrected residual median={med:.4f}, "
                f"sigmaMAD={mad:.4f}, N={n}"
            )

    return corr, matches


def add_corrected_t01_columns(cat: pd.DataFrame, corr: dict[str, float]) -> pd.DataFrame:
    out = cat.copy()
    for b, dcorr in corr.items():
        for prefix in [f"mag_{b}_AB", f"mag_aper_{b}_AB", f"mag_psf_{b}_AB"]:
            if prefix in out.columns:
                out[f"{prefix}_corr"] = finite_array(out[prefix]) + dcorr
    return out

def standardize_field(cat: pd.DataFrame, field: str, corr: dict[str, float] | None = None) -> pd.DataFrame:
    rows = pd.DataFrame()
    rows["source_id_input"] = cat["source_id"] if "source_id" in cat.columns else np.arange(len(cat))
    rows["field"] = field
    rows["ra_deg"] = finite_array(cat["ra_deg"])
    rows["dec_deg"] = finite_array(cat["dec_deg"])
    rows["x_ref"] = finite_array(cat["x_ref"]) if "x_ref" in cat.columns else np.nan
    rows["y_ref"] = finite_array(cat["y_ref"]) if "y_ref" in cat.columns else np.nan
    rows["in_slit_region"] = cat["in_slit_region"].astype(bool).to_numpy() if "in_slit_region" in cat.columns else False

    for b in ["r", "i", "z"]:
        mag_col = f"mag_{b}_AB_corr" if f"mag_{b}_AB_corr" in cat.columns else f"mag_{b}_AB"
        aper_col = f"mag_aper_{b}_AB_corr" if f"mag_aper_{b}_AB_corr" in cat.columns else f"mag_aper_{b}_AB"
        psf_col = f"mag_psf_{b}_AB_corr" if f"mag_psf_{b}_AB_corr" in cat.columns else f"mag_psf_{b}_AB"

        rows[f"mag_{b}_AB"] = finite_array(cat[mag_col]) if mag_col in cat.columns else np.nan
        rows[f"magerr_{b}_AB"] = finite_array(cat[f"magerr_{b}_AB"]) if f"magerr_{b}_AB" in cat.columns else np.nan
        rows[f"mag_aper_{b}_AB"] = finite_array(cat[aper_col]) if aper_col in cat.columns else np.nan
        rows[f"mag_psf_{b}_AB"] = finite_array(cat[psf_col]) if psf_col in cat.columns else np.nan
        rows[f"good_psf_{b}"] = cat[f"good_psf_{b}"].astype(bool).to_numpy() if f"good_psf_{b}" in cat.columns else False
        rows[f"mag_source_{b}"] = cat[f"mag_source_{b}"].astype(str).to_numpy() if f"mag_source_{b}" in cat.columns else ""

    rows["color_r_i_AB"] = rows["mag_r_AB"] - rows["mag_i_AB"]
    return rows

def clean_bool(s):
    return s.fillna(False).astype(bool)

def robust_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def measurement_quality_score(df: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(df), dtype=float)
    score += np.where(df["in_slit_region"].astype(bool).to_numpy(), 100.0, 0.0)

    for b in ["r", "i"]:
        gcol = f"good_psf_{b}"
        if gcol in df.columns:
            score += np.where(df[gcol].astype(bool).to_numpy(), 0.0, 10.0)

    err_terms = []
    for b in ["r", "i"]:
        ecol = f"magerr_{b}_AB"
        if ecol in df.columns:
            e = finite_array(df[ecol])
            err_terms.append(np.where(np.isfinite(e), e, 9.9))
    if err_terms:
        score += np.nanmean(np.vstack(err_terms), axis=0)

    return score


def build_bar_loss_table(matches: pd.DataFrame, corr: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, r in matches.iterrows():
        bar0 = bool(r.get("in_slit_region_T00", False))
        bar1 = bool(r.get("in_slit_region_T01", False))

        if bar0 == bar1:
            continue

        barred = "T00" if bar0 else "T01"
        clear = "T01" if bar0 else "T00"

        row = {
            "source_id_T00": r.get("source_id_T00", ""),
            "source_id_T01": r.get("source_id_T01", ""),
            "ra_deg": np.nanmean([r.get("ra_deg_T00", np.nan), r.get("ra_deg_T01", np.nan)]),
            "dec_deg": np.nanmean([r.get("dec_deg_T00", np.nan), r.get("dec_deg_T01", np.nan)]),
            "match_sep_arcsec": r.get("match_sep_arcsec", np.nan),
            "barred_field": barred,
            "clear_field": clear,
        }

        for b in ["r", "i", "z"]:
            mb = r.get(f"mag_{b}_AB_{barred}", np.nan)
            mc = r.get(f"mag_{b}_AB_{clear}", np.nan)

            # Apply T01 relative correction before comparing fields.
            if barred == "T01" and np.isfinite(mb):
                mb = mb + corr.get(b, 0.0)
            if clear == "T01" and np.isfinite(mc):
                mc = mc + corr.get(b, 0.0)

            row[f"mag_{b}_barred_AB"] = mb
            row[f"mag_{b}_clear_AB"] = mc
            row[f"bar_loss_{b}_mag"] = mb - mc if np.isfinite(mb) and np.isfinite(mc) else np.nan
            row[f"flux_transmission_{b}"] = 10.0 ** (-0.4 * row[f"bar_loss_{b}_mag"]) if np.isfinite(row[f"bar_loss_{b}_mag"]) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_merged_catalog(t00: pd.DataFrame, t01: pd.DataFrame, matches: pd.DataFrame, corr: dict[str, float]) -> pd.DataFrame:
    t01c = add_corrected_t01_columns(t01, corr)

    s00 = standardize_field(t00, "T00")
    s01 = standardize_field(t01c, "T01", corr=corr)

    matched_ids_00 = set(matches["source_id_T00"].astype(str)) if "source_id_T00" in matches.columns else set()
    matched_ids_01 = set(matches["source_id_T01"].astype(str)) if "source_id_T01" in matches.columns else set()

    merged_rows = []

    for k, r in matches.iterrows():
        sid0 = str(r.get("source_id_T00", ""))
        sid1 = str(r.get("source_id_T01", ""))

        row00 = s00[s00["source_id_input"].astype(str) == sid0]
        row01 = s01[s01["source_id_input"].astype(str) == sid1]
        if len(row00) == 0 or len(row01) == 0:
            continue

        pair = pd.concat([row00, row01], ignore_index=True)
        pair = pair.copy()
        pair["score"] = measurement_quality_score(pair)
        best = pair.iloc[int(np.nanargmin(pair["score"].to_numpy()))].copy()

        out = {
            "merged_id": f"M{k+1:05d}",
            "source_type": "matched",
            "preferred_field": best["field"],
            "source_id_T00": sid0,
            "source_id_T01": sid1,
            "match_sep_arcsec": r.get("match_sep_arcsec", np.nan),
            "ra_deg": np.nanmean([r.get("ra_deg_T00", np.nan), r.get("ra_deg_T01", np.nan)]),
            "dec_deg": np.nanmean([r.get("dec_deg_T00", np.nan), r.get("dec_deg_T01", np.nan)]),
            "in_slit_region_T00": bool(r.get("in_slit_region_T00", False)),
            "in_slit_region_T01": bool(r.get("in_slit_region_T01", False)),
        }

        for b in ["r", "i", "z"]:
            out[f"mag_{b}_AB"] = best.get(f"mag_{b}_AB", np.nan)
            out[f"magerr_{b}_AB"] = best.get(f"magerr_{b}_AB", np.nan)
            out[f"mag_source_{b}"] = best.get(f"mag_source_{b}", "")
            out[f"good_psf_{b}"] = best.get(f"good_psf_{b}", False)

            m00 = row00.iloc[0].get(f"mag_{b}_AB", np.nan)
            e00 = row00.iloc[0].get(f"magerr_{b}_AB", np.nan)
            m01 = row01.iloc[0].get(f"mag_{b}_AB", np.nan)
            e01 = row01.iloc[0].get(f"magerr_{b}_AB", np.nan)

            out[f"mag_{b}_AB_T00"] = m00
            out[f"magerr_{b}_AB_T00"] = e00
            out[f"mag_{b}_AB_T01_corr"] = m01
            out[f"magerr_{b}_AB_T01"] = e01
            out[f"dmag_{b}_T01corr_minus_T00"] = m01 - m00 if np.isfinite(m00) and np.isfinite(m01) else np.nan

        out["color_r_i_AB"] = out["mag_r_AB"] - out["mag_i_AB"] if np.isfinite(out.get("mag_r_AB", np.nan)) and np.isfinite(out.get("mag_i_AB", np.nan)) else np.nan
        merged_rows.append(out)

    # Unmatched rows.
    s00_un = s00[~s00["source_id_input"].astype(str).isin(matched_ids_00)].copy()
    s01_un = s01[~s01["source_id_input"].astype(str).isin(matched_ids_01)].copy()

    for df_un, field in [(s00_un, "T00"), (s01_un, "T01")]:
        for _, r in df_un.iterrows():
            out = {
                "merged_id": f"{field}_{r['source_id_input']}",
                "source_type": "single",
                "preferred_field": field,
                "source_id_T00": r["source_id_input"] if field == "T00" else "",
                "source_id_T01": r["source_id_input"] if field == "T01" else "",
                "match_sep_arcsec": np.nan,
                "ra_deg": r["ra_deg"],
                "dec_deg": r["dec_deg"],
                "in_slit_region_T00": bool(r["in_slit_region"]) if field == "T00" else False,
                "in_slit_region_T01": bool(r["in_slit_region"]) if field == "T01" else False,
            }
            for b in ["r", "i", "z"]:
                out[f"mag_{b}_AB"] = r.get(f"mag_{b}_AB", np.nan)
                out[f"magerr_{b}_AB"] = r.get(f"magerr_{b}_AB", np.nan)
                out[f"mag_source_{b}"] = r.get(f"mag_source_{b}", "")
                out[f"good_psf_{b}"] = r.get(f"good_psf_{b}", False)

                out[f"mag_{b}_AB_T00"] = r.get(f"mag_{b}_AB", np.nan) if field == "T00" else np.nan
                out[f"magerr_{b}_AB_T00"] = r.get(f"magerr_{b}_AB", np.nan) if field == "T00" else np.nan
                out[f"mag_{b}_AB_T01_corr"] = r.get(f"mag_{b}_AB", np.nan) if field == "T01" else np.nan
                out[f"magerr_{b}_AB_T01"] = r.get(f"magerr_{b}_AB", np.nan) if field == "T01" else np.nan
                out[f"dmag_{b}_T01corr_minus_T00"] = np.nan

            out["color_r_i_AB"] = out["mag_r_AB"] - out["mag_i_AB"] if np.isfinite(out.get("mag_r_AB", np.nan)) and np.isfinite(out.get("mag_i_AB", np.nan)) else np.nan
            merged_rows.append(out)

    merged = pd.DataFrame(merged_rows)
    if len(merged):
        merged = merged.sort_values(["ra_deg", "dec_deg"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SISI T00/T01 final photometry catalogs.")
    parser.add_argument("--root", default="products", help="SISI products root")
    parser.add_argument("--target", default="Dolidze25")
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--corr-r", type=float, default=None,
                        help="Override T01->T00 r correction. Default: derive from clipped overlap.")
    parser.add_argument("--corr-i", type=float, default=None,
                        help="Override T01->T00 i correction. Default: derive from clipped overlap.")
    parser.add_argument("--corr-z", type=float, default=DEFAULT_REL_CORR["z"])
    parser.add_argument("--overlap-max-resid", type=float, default=0.50,
                        help="Hard absolute residual cut [mag] for overlap photcal stars.")
    parser.add_argument("--overlap-clip-nsigma", type=float, default=3.0,
                        help="SigmaMAD clipping threshold for overlap correction.")
    parser.add_argument("--overlap-min-stars", type=int, default=8,
                        help="Minimum stars needed to derive an overlap correction.")
    args = parser.parse_args()

    root = Path(args.root)
    t00_dir = root / f"{args.target}_T00"
    t01_dir = root / f"{args.target}_T01"
    out_dir = root / f"{args.target}_merged_photometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_bands = set()
    corr = {"r": DEFAULT_REL_CORR["r"], "i": DEFAULT_REL_CORR["i"], "z": args.corr_z}
    if args.corr_r is None:
        auto_bands.add("r")
    else:
        corr["r"] = args.corr_r
    if args.corr_i is None:
        auto_bands.add("i")
    else:
        corr["i"] = args.corr_i

    t00_csv = find_catalog(t00_dir)
    t01_csv = find_catalog(t01_dir)

    print("Reading:")
    print("  T00:", t00_csv)
    print("  T01:", t01_csv)

    t00 = pd.read_csv(t00_csv)
    t01 = pd.read_csv(t01_csv)

    matches = match_catalogs(t00, t01, radius_arcsec=args.match_radius)

    # Mark overlap pairs unsuitable for field-to-field residual statistics.
    matches["bad_overlap"] = False

    # Reject stars on SAMOS slit/bar in either field.
    for field in ["T00", "T01"]:
        col = f"in_slit_region_{field}"
        if col in matches.columns:
            matches["bad_overlap"] |= matches[col].fillna(False).astype(bool)

    # Reject bad PSF in either field, either band.
    for field in ["T00", "T01"]:
        for b in ["r", "i"]:
            col = f"good_psf_{b}_{field}"
            if col in matches.columns:
                matches["bad_overlap"] |= ~matches[col].fillna(False).astype(bool)

    # Derive/check T01->T00 corrections and finalize the clean overlap mask.
    corr, matches = apply_overlap_quality_and_corrections(
        matches,
        corr,
        auto_bands,
        max_resid=args.overlap_max_resid,
        nsigma=args.overlap_clip_nsigma,
        min_n=args.overlap_min_stars,
    )

    for b, dcorr in corr.items():
        col = f"mag_{b}_AB_T01"
        if col in matches.columns:
            matches[f"mag_{b}_AB_T01_corr"] = finite_array(matches[col]) + dcorr
        col0 = f"mag_{b}_AB_T00"
        col1c = f"mag_{b}_AB_T01_corr"
        if col0 in matches.columns and col1c in matches.columns:
            matches[f"dmag_{b}_T01corr_minus_T00"] = finite_array(matches[col1c]) - finite_array(matches[col0])

    bar_loss = build_bar_loss_table(matches, corr=corr)
    merged = build_merged_catalog(t00, t01, matches, corr=corr)

    matches_out = out_dir / "SISI_overlap_matches_all.csv"
    bar_out = out_dir / "SISI_overlap_bar_loss.csv"
    merged_out = out_dir / "SISI_merged_photometry.csv"

    matches.to_csv(matches_out, index=False)
    bar_loss.to_csv(bar_out, index=False)
    merged.to_csv(merged_out, index=False)

    print()
    print("Wrote:")
    print(" ", matches_out)
    print(" ", bar_out)
    print(" ", merged_out)

    print()
    print("Summary:")
    print(f"  matched T00/T01 sources : {len(matches)}")
    print(f"  bar-loss pairs          : {len(bar_loss)}")
    print(f"  merged catalog rows     : {len(merged)}")

    print("Relative T01 corrections:")
    print(f"  r: +{corr['r']:.3f} mag")
    print(f"  i: +{corr['i']:.3f} mag")
    print(f"  z: +{corr['z']:.3f} mag")


    for b in ["r", "i", "z"]:
        col = f"dmag_{b}_T01corr_minus_T00"
        if col in matches.columns:
            good = matches["use_overlap_photcal"].fillna(False).astype(bool)
            vals = pd.to_numeric(matches.loc[good, col], errors="coerce")
            med, mad, n = robust_median_mad(vals)
            nrej = int((~good & np.isfinite(pd.to_numeric(matches[col], errors="coerce"))).sum())
            print(
                f"  clean overlap, {b}: median T01-T00={med:.4f}, "
                f"sigmaMAD={mad:.4f}, N={n}, rejected={nrej}"
            )

    for b in ["r", "i", "z"]:
        col = f"bar_loss_{b}_mag"
        if col in bar_loss.columns:
            med, mad, n = robust_median_mad(bar_loss[col])
            print(f"  bar loss {b}: median={med:.4f} mag, sigmaMAD={mad:.4f}, N={n}")


if __name__ == "__main__":
    main()
