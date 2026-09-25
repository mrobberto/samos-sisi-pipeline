#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11 QC: SkyMapper photometric calibration.

Read-only QC for:
    pipeline/sisi/step11_photcal/step11_skymapper_calibrate.py

This QC script summarizes the final Step11 global photometric calibration.
It reads the SkyMapper matches, zero-point table, and calibrated Step09 field
catalog written by Step11.  It does not modify any pipeline products.

The report focuses on the operational/global calibration only.  Left/right
amplifier balancing is handled upstream by Step09 forced-centroid calibration
QC and is not expanded into separate pages here.

Run:
    PYTHONPATH=. SISI_ACTIVE_REDUCTION=config.reductions.sisi_Dolidze25_T00 \
    python qc/sisi/step11/qc_step11_skymapper_photcal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config
config.build_sisi_paths()


BANDS = list(getattr(config, "SISI_SCIENCE_FRAMES", {}).keys()) or ["r", "i", "z"]


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def finite_array(x):
    return np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.nanmedian(x))
    sigmad = float(1.4826 * np.nanmedian(np.abs(x - med)))
    return med, sigmad


def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(x * x)))


def quality_label(sigmad_resid):
    if not np.isfinite(sigmad_resid):
        return "unknown"
    if sigmad_resid < 0.03:
        return "excellent"
    if sigmad_resid < 0.05:
        return "very good"
    if sigmad_resid < 0.10:
        return "acceptable"
    return "suspicious"


def overall_quality(stats_list):
    qualities = [s["quality"] for s in stats_list if s is not None]
    if not qualities:
        return "UNKNOWN"
    if any(q == "suspicious" for q in qualities):
        return "SUSPICIOUS"
    if all(q in ("excellent", "very good", "acceptable") for q in qualities):
        return "ACCEPTABLE"
    return "UNKNOWN"


def count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(sum(1 for _ in open(path, "r")) - 1, 0)
    except Exception:
        return None


def get_xy(m: pd.DataFrame):
    x_col = first_existing(
        m.columns,
        ["x_detect_i", "x_detect_r", "x_i", "x_r", "x", "x_ref", "X", "x_pix"],
    )
    y_col = first_existing(
        m.columns,
        ["y_detect_i", "y_detect_r", "y_i", "y_r", "y", "y_ref", "Y", "y_pix"],
    )
    x = finite_array(m[x_col]) if x_col else np.full(len(m), np.nan)
    y = finite_array(m[y_col]) if y_col else np.full(len(m), np.nan)
    split = float(getattr(config, "SISI_AMP_SPLIT_COL", 528.0))
    return x, y, split, x_col, y_col


def get_band_stats(m: pd.DataFrame, z: pd.DataFrame, band: str):
    """Return global photometric calibration diagnostics for one band."""
    inst_col = first_existing(
        m.columns,
        [
            f"mag_psf_inst_{band}",
            f"mag_inst_{band}",
            f"mag_{band}",
            f"{band}_mag",
            f"inst_mag_{band}",
            f"{band}_inst",
            f"{band}_apmag",
            f"apmag_{band}",
        ],
    )
    sm_col = f"sm_{band}"
    use_col = f"use_zp_{band}"

    if inst_col is None or sm_col not in m.columns:
        return None

    smag = finite_array(m[sm_col])
    inst = finite_array(m[inst_col])
    delta = smag - inst

    use = (
        m[use_col].astype(bool).to_numpy().copy()
        if use_col in m.columns
        else np.isfinite(delta).copy()
    )
    use &= np.isfinite(delta)

    zp_row = z[z["band"].astype(str) == band]
    if len(zp_row):
        zp = float(zp_row.iloc[0].get("zp", np.nan))
        zp_adus = float(zp_row.iloc[0].get("zp_adus", np.nan))
        exptime_s = float(zp_row.iloc[0].get("exptime_s", np.nan))
        zp_err = float(zp_row.iloc[0].get("zp_err", np.nan))
    else:
        zp, scatter = robust_median_mad(delta[use])
        exptime_s = np.nan
        zp_adus = np.nan
        zp_err = scatter / np.sqrt(use.sum()) if use.sum() else np.nan

    residual = delta - zp
    residual_used = residual[use]
    med_resid, sigmad_resid = robust_median_mad(residual_used)
    rms_resid = rms(residual_used)

    return dict(
        band=band,
        inst_col=inst_col,
        sm_col=sm_col,
        smag=smag,
        inst=inst,
        delta=delta,
        use=use,
        zp=zp,
        zp_adus=zp_adus,
        exptime_s=exptime_s,
        zp_err=zp_err,
        n_total=int(np.isfinite(delta).sum()),
        n_used=int(use.sum()),
        residual=residual,
        med_resid=med_resid,
        sigmad_resid=sigmad_resid,
        rms_resid=rms_resid,
        quality=quality_label(sigmad_resid),
    )


def write_summary(qc_dir: Path, out_dir: Path, m: pd.DataFrame, z: pd.DataFrame, c: pd.DataFrame):
    summary_path = qc_dir / "SISI_skymapper_calibration_summary.txt"

    step09_dir = Path(config.SISI_REDUCED) / "09_photometry"
    sky_query = out_dir / "SISI_skymapper_query.csv"
    all_csv = step09_dir / "SISI_detected_all_photometry.csv"
    field_csv = step09_dir / "SISI_detected_field_photometry.csv"
    slit_csv = step09_dir / "SISI_detected_in_slit_photometry.csv"

    in_slit = (
        m["in_slit_region"].astype(bool).to_numpy()
        if "in_slit_region" in m.columns
        else np.zeros(len(m), dtype=bool)
    )

    stats_list = [get_band_stats(m, z, band) for band in BANDS]
    verdict = overall_quality(stats_list)

    lines = []
    lines.append("SISI / SkyMapper photometric calibration summary")
    lines.append("=" * 54)
    lines.append("")
    lines.append(f"Target                : {getattr(config, 'SISI_TARGET', 'UNKNOWN')}")
    lines.append(f"Field                 : {getattr(config, 'SISI_FIELD', 'UNKNOWN')}")
    lines.append(f"SISI_REDUCED          : {config.SISI_REDUCED}")
    lines.append(f"Photometric status    : {verdict}")
    lines.append("")
    lines.append("Catalog / detection counts")
    lines.append("-" * 28)
    lines.append(f"SkyMapper queried      : {count_rows(sky_query)}")
    lines.append(f"SISI detections all    : {count_rows(all_csv)}")
    lines.append(f"SISI detections field  : {count_rows(field_csv)}")
    lines.append(f"SISI detections in slit: {count_rows(slit_csv)}")
    lines.append(f"Matched calibrators    : {len(m)}")
    lines.append(f"Matched in slit/bar    : {int(in_slit.sum())}")
    lines.append(f"Calibrated field rows  : {len(c)}")
    lines.append("")
    lines.append("Zero-point diagnostics")
    lines.append("-" * 28)

    for band, stats in zip(BANDS, stats_list):
        if stats is None:
            lines.append(f"Band {band}: not available")
            continue

        lines.append(f"Band {band}  GLOBAL / pipeline ZP")
        lines.append(f"  instrumental column     : {stats['inst_col']}")
        lines.append(f"  SkyMapper column        : {stats['sm_col']}")
        lines.append(f"  N total / N used        : {stats['n_total']} / {stats['n_used']}")
        lines.append(f"  zero point ADU          : {stats['zp']:.5f} mag")
        lines.append(f"  zero point ADU/s        : {stats['zp_adus']:.5f} mag")
        lines.append(f"  zero point error        : {stats['zp_err']:.5f} mag")
        lines.append(f"  residual median         : {stats['med_resid']:.5f} mag")
        lines.append(f"  residual sigma_MAD      : {stats['sigmad_resid']:.5f} mag")
        lines.append(f"  residual RMS            : {stats['rms_resid']:.5f} mag")
        lines.append(f"  quality                 : {stats['quality']}")
        lines.append("")

    lines.append("Interpretation guide")
    lines.append("-" * 28)
    lines.append("The global zero point is the operational Step11 calibration.")
    lines.append("Rejected points are kept in the QC plots for context but are not used in the zero-point solution.")
    lines.append("Left/right amplifier balancing is tested upstream in Step09 forced-centroid QC.")
    lines.append("")

    summary_path.write_text("\n".join(lines))
    print(f"Wrote summary: {summary_path}")
    return summary_path, lines, verdict


def plot_zp_distribution(stats, qc_dir: Path, pdf: PdfPages | None = None):
    band = stats["band"]
    delta = stats["delta"]
    use = stats["use"]
    exptime_s = stats.get("exptime_s", np.nan)

    if np.isfinite(exptime_s) and exptime_s > 0:
        delta_plot = delta - 2.5 * np.log10(exptime_s)
        zp_plot = stats["zp_adus"]
        xlabel = f"SkyMapper {band} - SISI instrumental {band} [ADU/s mag]"
    else:
        delta_plot = delta
        zp_plot = stats["zp"]
        xlabel = f"SkyMapper {band} - SISI instrumental {band} [ADU mag]"

    valid = np.isfinite(delta_plot)
    if valid.sum() == 0:
        return

    lo, hi = np.nanpercentile(delta_plot[valid], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(delta_plot[valid]), np.nanmax(delta_plot[valid])

    pad = 0.15 * max(hi - lo, 0.1)
    bins = np.linspace(lo - pad, hi + pad, 40)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta_plot[valid & ~use], bins=bins, alpha=0.35, label="rejected")
    ax.hist(delta_plot[use], bins=bins, alpha=0.75, label="used")
    ax.axvline(zp_plot, linestyle="--", linewidth=2, label=f"ZP={zp_plot:.4f}")
    ax.axvline(zp_plot - stats["sigmad_resid"], linestyle=":", linewidth=1)
    ax.axvline(zp_plot + stats["sigmad_resid"], linestyle=":", linewidth=1,
               label=f"resid σMAD={stats['sigmad_resid']:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of matched sources")
    ax.set_title(f"Zero-point distribution: {band}")
    ax.legend()

    png = qc_dir / f"zp_distribution_{band}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def plot_residual_vs_magnitude(stats, pdf: PdfPages):
    band = stats["band"]
    smag = stats["smag"]
    use = stats["use"]
    residual = stats["residual"]

    ok = np.isfinite(smag) & np.isfinite(residual)
    if ok.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(smag[ok & ~use], residual[ok & ~use], s=12, alpha=0.35, label="rejected")
    ax.scatter(smag[ok & use], residual[ok & use], s=18, alpha=0.85, label="used")
    ax.axhline(0.0, linestyle="--")
    ax.axhline(stats["sigmad_resid"], linestyle=":")
    ax.axhline(-stats["sigmad_resid"], linestyle=":")
    ax.set_xlabel(f"SkyMapper {band}")
    ax.set_ylabel(f"SkyMapper {band} - calibrated SISI {band}")
    ax.set_title(f"Post-calibration residuals: {band} ({stats['quality']})")
    ax.invert_xaxis()
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_color_term(stats, m: pd.DataFrame, pdf: PdfPages):
    if "sm_r" not in m.columns or "sm_i" not in m.columns:
        return

    band = stats["band"]
    color = finite_array(m["sm_r"]) - finite_array(m["sm_i"])
    residual = stats["residual"]
    use = stats["use"]

    ok = np.isfinite(color) & np.isfinite(residual)
    if ok.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(color[ok & ~use], residual[ok & ~use], s=12, alpha=0.35, label="rejected")
    ax.scatter(color[ok & use], residual[ok & use], s=18, alpha=0.85, label="used")
    ax.axhline(0.0, linestyle="--")

    slope_text = ""
    fit_ok = ok & use
    if fit_ok.sum() >= 5:
        p = np.polyfit(color[fit_ok], residual[fit_ok], 1)
        xs = np.linspace(np.nanmin(color[fit_ok]), np.nanmax(color[fit_ok]), 100)
        ax.plot(xs, p[0] * xs + p[1], linestyle="-", linewidth=1.5,
                label=f"slope={p[0]:+.3f} mag/mag")
        slope_text = f"; slope={p[0]:+.3f} mag/mag"

    ax.set_xlabel("SkyMapper r - i")
    ax.set_ylabel(f"Residual after ZP: {band}")
    ax.set_title(f"Color-term diagnostic: {band}{slope_text}")
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_skymapper_vs_sisi(stats, qc_dir: Path, pdf: PdfPages):
    band = stats["band"]
    smag = stats["smag"]
    sisi_cal = stats["inst"] + stats["zp"]
    use = stats["use"]

    ok = np.isfinite(smag) & np.isfinite(sisi_cal)
    if ok.sum() == 0:
        return

    lo = np.nanmin(np.r_[smag[ok], sisi_cal[ok]])
    hi = np.nanmax(np.r_[smag[ok], sisi_cal[ok]])
    pad = 0.05 * max(hi - lo, 1.0)

    resid = smag - sisi_cal
    med, sig = robust_median_mad(resid[ok & use])
    rr = rms(resid[ok & use])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sisi_cal[ok & ~use], smag[ok & ~use], s=12, alpha=0.25, label="rejected")
    ax.scatter(sisi_cal[ok & use], smag[ok & use], s=22, alpha=0.85, label="used")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.5, label="1:1")
    ax.set_xlim(hi + pad, lo - pad)
    ax.set_ylim(hi + pad, lo - pad)
    ax.set_xlabel(f"Calibrated SISI {band}")
    ax.set_ylabel(f"SkyMapper {band}")
    ax.set_title(f"SkyMapper vs calibrated SISI: {band}\nmedian resid={med:+.4f}, σMAD={sig:.4f}, RMS={rr:.4f}")
    ax.legend()
    ax.grid(alpha=0.25)

    png = qc_dir / f"qc_step11_skymapper_vs_sisi_{band}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def plot_residual_vs_position(stats, pdf: PdfPages):
    x, y, split, x_col, y_col = get_xy(pd.DataFrame({}))
    # This placeholder is deliberately unused.  The actual map is handled by
    # plot_detector_residual_map(), which has access to the full match table.
    return


def plot_residual_vs_xy(stats, m: pd.DataFrame, pdf: PdfPages):
    x, y, split, x_col, y_col = get_xy(m)
    if x_col is None or y_col is None:
        return

    band = stats["band"]
    use = stats["use"]
    residual = stats["residual"]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
    if (ok & use).sum() == 0:
        return

    clip = max(0.3, 3.0 * stats["sigmad_resid"]) if np.isfinite(stats["sigmad_resid"]) else 0.5

    for coord_name, coord, xlabel in [("x", x, "x [pix]"), ("y", y, "y [pix]")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(coord[ok & ~use], residual[ok & ~use], s=14, alpha=0.25, label="rejected")
        ax.scatter(coord[ok & use], residual[ok & use], s=25, alpha=0.85, label="used")
        ax.axhline(0.0, linestyle="--")
        ax.axhline(stats["sigmad_resid"], linestyle=":")
        ax.axhline(-stats["sigmad_resid"], linestyle=":")
        if coord_name == "x":
            ax.axvline(split, color="k", linestyle="--", linewidth=1)
        ax.set_ylim(-clip, clip)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{band} residual after ZP [mag]")
        ax.set_title(f"Residual vs {coord_name}: {band}")
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_detector_residual_map(stats, m: pd.DataFrame, qc_dir: Path, pdf: PdfPages):
    x, y, split, x_col, y_col = get_xy(m)
    if x_col is None or y_col is None:
        return

    band = stats["band"]
    use = stats["use"]
    residual = stats["residual"]
    smag = stats["smag"]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
    rej = ok & ~use
    used = ok & use

    if used.sum() == 0:
        return

    clip = max(0.3, 3.0 * stats["sigmad_resid"])
    size = 20 + 60 * np.clip((20.0 - smag[used]) / 2.0, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[rej], y[rej], s=20, alpha=0.30, marker="x", label="rejected")
    sc = ax.scatter(
        x[used], y[used],
        c=residual[used],
        s=size,
        cmap="coolwarm",
        vmin=-clip,
        vmax=+clip,
        edgecolors="k",
        linewidths=0.3,
        label="used",
    )
    ax.axvline(split, color="k", linestyle="--", linewidth=1)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(f"{band} residual after global ZP [mag]")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"Residual after global photometric calibration: {band}")
    ax.legend(loc="best")
    pdf.savefig(fig, bbox_inches="tight")

    png = qc_dir / f"qc_step11_match_map_{band}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def plot_calibrated_cmd(c: pd.DataFrame, qc_dir: Path, pdf: PdfPages):
    if "mag_r_skymapper" not in c.columns or "mag_i_skymapper" not in c.columns:
        return

    color = finite_array(c["mag_r_skymapper"]) - finite_array(c["mag_i_skymapper"])
    mag = finite_array(c["mag_i_skymapper"])
    ok = np.isfinite(color) & np.isfinite(mag)

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(color[ok], mag[ok], s=12, alpha=0.75)
    ax.set_xlabel("r - i calibrated")
    ax.set_ylabel("i calibrated")
    ax.set_title("SISI calibrated CMD")
    ax.invert_yaxis()
    pdf.savefig(fig, bbox_inches="tight")

    png = qc_dir / "qc_step11_cmd_r_i_calibrated.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


def plot_match_sky_map(m: pd.DataFrame, pdf: PdfPages):
    ra_col = first_existing(m.columns, ["ra_deg", "RA_deg", "ra", "RA"])
    dec_col = first_existing(m.columns, ["dec_deg", "DEC_deg", "dec", "DEC"])
    if not (ra_col and dec_col):
        return

    ra = finite_array(m[ra_col])
    dec = finite_array(m[dec_col])
    ok = np.isfinite(ra) & np.isfinite(dec)
    if ok.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ra[ok], dec[ok], s=15, alpha=0.8)
    ax.set_xlabel("RA deg")
    ax.set_ylabel("Dec deg")
    ax.set_title("SkyMapper matched calibrators")
    ax.invert_xaxis()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(config.SISI_REDUCED) / "11_photcal"
    qc_dir = Path(config.SISI_ST11_PHOTCAL_QC)
    qc_dir.mkdir(parents=True, exist_ok=True)

    match_csv = out_dir / "SISI_skymapper_matches.csv"
    zp_csv = out_dir / "SISI_skymapper_zero_points.csv"
    cal_csv = out_dir / "SISI_detected_field_photometry_skymapper_calibrated.csv"

    if not match_csv.exists():
        raise FileNotFoundError(f"Missing {match_csv}")
    if not zp_csv.exists():
        raise FileNotFoundError(f"Missing {zp_csv}")
    if not cal_csv.exists():
        raise FileNotFoundError(f"Missing {cal_csv}")

    m = pd.read_csv(match_csv)
    z = pd.read_csv(zp_csv)
    c = pd.read_csv(cal_csv)

    summary_path, summary_lines, verdict = write_summary(qc_dir, out_dir, m, z, c)
    pdf_path = qc_dir / "qc_step11_skymapper_photcal.pdf"

    with PdfPages(pdf_path) as pdf:
        # Text summary page.
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(summary_lines), va="top",
                family="monospace", fontsize=7.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Compact zero-point summary page.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        lines = ["SkyMapper photometric calibration", "", f"Overall status: {verdict}", ""]
        for b in BANDS:
            g = get_band_stats(m, z, b)
            if g is None:
                continue
            lines.append(
                f"{b}: ZP={g['zp']:.4f} ± {g['zp_err']:.4f}, "
                f"resid_MAD={g['sigmad_resid']:.4f}, N={g['n_used']}/{g['n_total']}"
            )
        ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Global band diagnostics only.
        for b in BANDS:
            stats = get_band_stats(m, z, b)
            if stats is None:
                continue

            plot_zp_distribution(stats, qc_dir, pdf=pdf)
            plot_residual_vs_magnitude(stats, pdf)
            plot_color_term(stats, m, pdf)
            plot_skymapper_vs_sisi(stats, qc_dir, pdf)
            plot_residual_vs_xy(stats, m, pdf)
            plot_detector_residual_map(stats, m, qc_dir, pdf)

        plot_calibrated_cmd(c, qc_dir, pdf)
        plot_match_sky_map(m, pdf)

    print(f"Wrote QC PDF: {pdf_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
