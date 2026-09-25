#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISI Step14 merged photometry QC.

This is a diagnostic QC for the final T00/T01 merged catalog.  It summarizes
catalog counts, plots the merged CMD and sky map, and inspects the overlap
residuals after the T01->T00 correction.

Compared with the older version, this script:

* auto-scales the merged CMD using the actual catalog percentiles;
* computes overlap statistics from the clean/use_overlap_photcal subset;
* plots all overlap matches classified by likely failure mode;
* highlights barred, non-PSF, aperture-fallback, and rejected overlap pairs;
* writes a table of large overlap residual outliers for inspection.

Run
---
PYTHONPATH=. python qc/sisi/step14/qc_step14_merged_photometry.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def finite_array(x):
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def robust_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return float(med), float(mad), int(len(x))


def bool_col(df: pd.DataFrame, name: str, default: bool = False) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=bool)
    return df[name].fillna(default).astype(bool).to_numpy()


def str_col(df: pd.DataFrame, name: str, default: str = "") -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=object)
    return df[name].fillna(default).astype(str).to_numpy()


def percentile_limits(x, lo=0.5, hi=99.5, pad_frac=0.07, hard=None):
    """Return padded percentile limits for finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if hard is not None:
        x = x[(x >= hard[0]) & (x <= hard[1])]
    if len(x) == 0:
        return None
    a, b = np.nanpercentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return None
    pad = pad_frac * (b - a)
    return a - pad, b + pad


def classify_overlap(matches: pd.DataFrame, band: str) -> np.ndarray:
    """Classify overlap rows for plotting diagnostics."""
    n = len(matches)
    cls = np.full(n, "clean/use", dtype=object)

    use = bool_col(matches, "use_overlap_photcal", default=False)
    cls[~use] = "rejected overlap"

    barred = bool_col(matches, "in_slit_region_T00") | bool_col(matches, "in_slit_region_T01")
    cls[barred] = "bar/slit"

    # Poor PSF in either field or either production band is a stronger reason
    # than generic rejection.
    bad_psf = np.zeros(n, dtype=bool)
    for fld in ["T00", "T01"]:
        for b in ["r", "i"]:
            col = f"good_psf_{b}_{fld}"
            if col in matches.columns:
                bad_psf |= ~bool_col(matches, col, default=False)
    cls[bad_psf & ~barred] = "bad PSF"

    # Aperture fallback in the plotted band.
    src0 = str_col(matches, f"mag_source_{band}_T00")
    src1 = str_col(matches, f"mag_source_{band}_T01")
    aper = (src0 == "aper") | (src1 == "aper")
    cls[aper & ~barred & ~bad_psf] = "aperture fallback"

    # Very loose matches should be flagged even if they otherwise pass.
    if "match_sep_arcsec" in matches.columns:
        sep = finite_array(matches["match_sep_arcsec"])
        loose = np.isfinite(sep) & (sep > 1.0)
        cls[loose & (cls == "clean/use")] = "loose match >1arcsec"

    return cls


def scatter_by_class(ax, x, y, cls, s=14):
    order = [
        "clean/use",
        "loose match >1arcsec",
        "aperture fallback",
        "bad PSF",
        "bar/slit",
        "rejected overlap",
    ]
    for label in order:
        use = (cls == label) & np.isfinite(x) & np.isfinite(y)
        if np.any(use):
            ax.scatter(x[use], y[use], s=s, alpha=0.65, label=f"{label} ({use.sum()})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="products")
    parser.add_argument("--target", default="Dolidze25")
    parser.add_argument("--outlier-threshold", type=float, default=0.5,
                        help="Absolute residual threshold for overlap outlier table [mag].")
    args = parser.parse_args()

    out_dir = Path(args.root) / f"{args.target}_merged_photometry"
    merged_csv = out_dir / "SISI_merged_photometry.csv"
    matches_csv = out_dir / "SISI_overlap_matches_all.csv"

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if not merged_csv.exists():
        raise FileNotFoundError(merged_csv)
    if not matches_csv.exists():
        raise FileNotFoundError(matches_csv)

    merged = pd.read_csv(merged_csv)
    matches = pd.read_csv(matches_csv)

    if "use_overlap_photcal" in matches.columns:
        matches_clean = matches[matches["use_overlap_photcal"].fillna(False).astype(bool)].copy()
    else:
        matches_clean = matches.copy()

    pdf_path = qc_dir / "qc_step14_merged_photometry.pdf"
    outlier_csv = qc_dir / "qc_step14_overlap_outliers.csv"

    # ------------------------------------------------------------------
    # Build outlier table for manual inspection.
    # ------------------------------------------------------------------
    outlier_mask = np.zeros(len(matches), dtype=bool)
    for b in ["r", "i", "z"]:
        col = f"dmag_{b}_T01corr_minus_T00"
        if col in matches.columns:
            d = finite_array(matches[col])
            outlier_mask |= np.isfinite(d) & (np.abs(d) > args.outlier_threshold)

    outlier_cols = [
        "source_id_T00", "source_id_T01", "match_sep_arcsec",
        "in_slit_region_T00", "in_slit_region_T01", "use_overlap_photcal",
    ]
    for b in ["r", "i", "z"]:
        outlier_cols += [
            f"mag_{b}_AB_T00", f"mag_{b}_AB_T01", f"mag_{b}_AB_T01_corr",
            f"dmag_{b}_T01corr_minus_T00",
            f"mag_source_{b}_T00", f"mag_source_{b}_T01",
            f"good_psf_{b}_T00", f"good_psf_{b}_T01",
        ]
    outlier_cols = [c for c in outlier_cols if c in matches.columns]
    matches.loc[outlier_mask, outlier_cols].to_csv(outlier_csv, index=False)

    with PdfPages(pdf_path) as pdf:
        # --------------------------------------------------------------
        # Summary page
        # --------------------------------------------------------------
        lines = [
            "SISI Step14 merged photometry QC",
            "=" * 42,
            "",
            f"Target        : {args.target}",
            f"Product dir   : {out_dir}",
            f"Merged rows   : {len(merged)}",
            f"Overlap rows  : {len(matches)}",
            f"Clean overlap : {len(matches_clean)}",
            f"Outliers |dmag|>{args.outlier_threshold:.2f}: {int(outlier_mask.sum())}",
            f"Outlier table : {outlier_csv}",
            "",
            "Merged catalog counts",
            "-" * 24,
        ]

        for col in ["source_type", "preferred_field"]:
            if col in merged.columns:
                lines.append(f"{col}:")
                for k, v in merged[col].value_counts(dropna=False).items():
                    lines.append(f"  {k}: {v}")
                lines.append("")

        lines.append("Overlap residuals after T01 correction")
        lines.append("clean/use_overlap_photcal subset")
        lines.append("-" * 42)
        for b in ["r", "i", "z"]:
            col = f"dmag_{b}_T01corr_minus_T00"
            if col in matches_clean.columns:
                med, mad, n = robust_stats(matches_clean[col])
                lines.append(f"{b}: median={med:.4f}, sigmaMAD={mad:.4f}, N={n}")

        lines.append("")
        lines.append("Overlap category counts")
        lines.append("-" * 24)
        cls_for_counts = classify_overlap(matches, "i")
        for k, v in pd.Series(cls_for_counts).value_counts().items():
            lines.append(f"  {k}: {v}")

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------------------------
        # Merged CMD, autoscaled.
        # --------------------------------------------------------------
        if "mag_r_AB" in merged.columns and "mag_i_AB" in merged.columns:
            r = finite_array(merged["mag_r_AB"])
            i = finite_array(merged["mag_i_AB"])
            color = r - i
            ok = np.isfinite(color) & np.isfinite(i)

            # Suppress pathological points only for axis calculation.
            axis_ok = ok & (color > -1.0) & (color < 4.0) & (i > 10.0) & (i < 25.0)
            xlim = percentile_limits(color[axis_ok], 0.5, 99.5, pad_frac=0.10)
            ylim = percentile_limits(i[axis_ok], 0.5, 99.5, pad_frac=0.08)

            fig, ax = plt.subplots(figsize=(6, 7))
            ax.scatter(color[ok], i[ok], s=7, alpha=0.45)
            ax.set_xlabel("r - i [AB]")
            ax.set_ylabel("i [AB]")
            ax.set_title("Merged CMD")
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(ylim[1], ylim[0])  # astronomical convention
            else:
                ax.invert_yaxis()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --------------------------------------------------------------
        # Sky map.
        # --------------------------------------------------------------
        if "ra_deg" in merged.columns and "dec_deg" in merged.columns:
            ra = finite_array(merged["ra_deg"])
            dec = finite_array(merged["dec_deg"])
            ok = np.isfinite(ra) & np.isfinite(dec)

            fig, ax = plt.subplots(figsize=(7, 6))
            if "preferred_field" in merged.columns:
                for fld in sorted(merged["preferred_field"].dropna().unique()):
                    use = ok & (merged["preferred_field"].astype(str).to_numpy() == str(fld))
                    ax.scatter(ra[use], dec[use], s=7, alpha=0.50, label=str(fld))
                ax.legend()
            else:
                ax.scatter(ra[ok], dec[ok], s=7, alpha=0.50)

            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
            ax.set_title("Merged catalog sky map")
            ax.invert_xaxis()
            ax.set_aspect("equal", adjustable="box")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --------------------------------------------------------------
        # Overlap residual diagnostics.
        # --------------------------------------------------------------
        for b in ["r", "i", "z"]:
            dcol = f"dmag_{b}_T01corr_minus_T00"
            mcol = f"mag_{b}_AB_T00"
            if dcol not in matches.columns:
                continue

            d = finite_array(matches[dcol])
            x = finite_array(matches[mcol]) if mcol in matches.columns else np.arange(len(d), dtype=float)
            sep = finite_array(matches["match_sep_arcsec"]) if "match_sep_arcsec" in matches.columns else np.full(len(d), np.nan)
            ok = np.isfinite(d) & np.isfinite(x)
            clean = bool_col(matches, "use_overlap_photcal", default=False)
            med, mad, n = robust_stats(d[ok & clean])
            cls = classify_overlap(matches, b)

            # Residual versus magnitude, all classes.
            fig, ax = plt.subplots(figsize=(6.6, 5.2))
            scatter_by_class(ax, x, d, cls, s=16)
            ax.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
            if np.isfinite(med):
                ax.axhline(med, linestyle=":", linewidth=1.2, label=f"clean median={med:.3f}")
                if np.isfinite(mad):
                    ax.axhline(med + mad, linestyle=":", linewidth=0.9)
                    ax.axhline(med - mad, linestyle=":", linewidth=0.9)
                    ax.axhline(med + 3 * mad, linestyle="--", linewidth=0.8)
                    ax.axhline(med - 3 * mad, linestyle="--", linewidth=0.8)
            ax.set_xlabel(f"T00 {b} [AB]" if mcol in matches.columns else "match index")
            ax.set_ylabel(f"T01 corr - T00 {b} [mag]")
            ax.set_title(f"Overlap residuals by class: {b}, clean N={n}")
            ax.invert_xaxis()
            ax.legend(fontsize=7, loc="best")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Residual versus match separation.
            if np.isfinite(sep).any():
                fig, ax = plt.subplots(figsize=(6.4, 5.0))
                scatter_by_class(ax, sep, d, cls, s=16)
                ax.axhline(0.0, linestyle="--", color="k", linewidth=1.0)
                ax.axvline(0.5, linestyle=":", linewidth=1.0, label="0.5 arcsec")
                ax.axvline(1.0, linestyle=":", linewidth=1.0, label="1.0 arcsec")
                ax.set_xlabel("match separation [arcsec]")
                ax.set_ylabel(f"T01 corr - T00 {b} [mag]")
                ax.set_title(f"Overlap residual versus separation: {b}")
                ax.legend(fontsize=7, loc="best")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Histogram: all vs clean/use.
            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            ax.hist(d[np.isfinite(d)], bins=40, alpha=0.35, label="all finite")
            ax.hist(d[np.isfinite(d) & clean], bins=30, alpha=0.70, label="clean/use")
            if np.isfinite(med):
                ax.axvline(0.0, linestyle="--", color="k")
                ax.axvline(med, linestyle=":", label=f"clean median={med:.3f}")
                ax.axvline(med + mad, linestyle=":")
                ax.axvline(med - mad, linestyle=":")
            ax.set_xlabel(f"T01 corr - T00 {b} [mag]")
            ax.set_ylabel("N")
            ax.set_title(f"Overlap residual distribution: {b}")
            ax.legend(fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {outlier_csv}")


if __name__ == "__main__":
    main()
