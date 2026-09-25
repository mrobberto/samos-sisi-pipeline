#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step13 QC: final calibrated SISI photometry.

Read:
    13_final_photometry/SISI_final_photometry.csv

Write:
    13_final_photometry_qc/qc_step13_final_photometry.pdf

Diagnostics:
    1. Text summary.
    2. CMD with representative error bars.
    3. PSF - aperture calibrated magnitude residuals versus magnitude.
    4. PSF - aperture residuals versus X and Y position.
    5. PSF - aperture residuals versus color.
    6. PSF qfit versus magnitude.
    7. Photometric error versus magnitude.
    8. Good/bad PSF source maps.

Run:
    SISI_ACTIVE_REDUCTION=config.reductions.sisi_Dolidze25_T01 \
    PYTHONPATH=. python qc/sisi/step13/qc_step13_final_photometry.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config


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


def maybe_series(df, name, default=False):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def add_binned_median(ax, x, y, nbins=10):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < nbins:
        return
    edges = np.nanpercentile(x[ok], np.linspace(0, 100, nbins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return

    xm, ym, ye = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = ok & (x >= lo) & (x <= hi)
        if use.sum() < 5:
            continue
        med, mad, _ = robust_stats(y[use])
        xm.append(np.nanmedian(x[use]))
        ym.append(med)
        ye.append(mad)

    if xm:
        ax.errorbar(xm, ym, yerr=ye, fmt="o-", linewidth=1.2, markersize=4, label="binned median")


def main():
    reduced = Path(config.SISI_REDUCED)
    in_csv = reduced / "13_final_photometry" / "SISI_final_photometry.csv"
    qc_dir = reduced / "13_final_photometry_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}")

    df = pd.read_csv(in_csv)

    bands = [b for b in getattr(config, "SISI_SCIENCE_FRAMES", {}).keys()]
    if not bands:
        bands = sorted(
            c.replace("mag_", "").replace("_AB", "")
            for c in df.columns
            if c.startswith("mag_") and c.endswith("_AB") and "aper" not in c and "psf" not in c
        )

    pdf_path = qc_dir / "qc_step13_final_photometry.pdf"

    with PdfPages(pdf_path) as pdf:
        # ------------------------------------------------------------
        # Summary page
        # ------------------------------------------------------------
        lines = [
            "SISI Step13 final calibrated photometry QC",
            "=" * 52,
            "",
            f"Target       : {getattr(config, 'SISI_TARGET', 'UNKNOWN')}",
            f"Field        : {getattr(config, 'SISI_FIELD', 'UNKNOWN')}",
            f"SISI_REDUCED : {reduced}",
            f"Input table  : {in_csv}",
            f"N rows       : {len(df)}",
            f"Bands        : {bands}",
            "",
            "Band summary",
            "-" * 24,
        ]

        for b in bands:
            mag_col = f"mag_{b}_AB"
            psf_col = f"mag_psf_{b}_AB"
            aper_col = f"mag_aper_{b}_AB"
            good_col = f"good_psf_{b}"
            qfit_col = f"psf_qfit_{b}"
            err_col = f"magerr_{b}_AB"
            zps_col = f"zp_scatter_{b}"

            if mag_col not in df.columns:
                lines.append(f"{b}: missing {mag_col}")
                continue

            mag = finite_array(df[mag_col])
            nmag = np.isfinite(mag).sum()
            lines.append(f"Band {b}")
            lines.append(f"  calibrated finite mags : {nmag}")

            if good_col in df.columns:
                good = df[good_col].astype(bool).to_numpy()
                lines.append(f"  good PSF              : {good.sum()} / {len(good)}")

            if psf_col in df.columns and aper_col in df.columns:
                dmag = finite_array(df[psf_col]) - finite_array(df[aper_col])
                med, mad, n = robust_stats(dmag)
                lines.append(f"  PSF-aper dmag         : median={med:.4f}, sigmaMAD={mad:.4f}, N={n}")

            if qfit_col in df.columns:
                q = finite_array(df[qfit_col])
                med, mad, n = robust_stats(q)
                lines.append(f"  qfit                  : median={med:.4g}, sigmaMAD={mad:.4g}, N={n}")

            if err_col in df.columns:
                e = finite_array(df[err_col])
                med, mad, n = robust_stats(e)
                lines.append(f"  total mag error        : median={med:.4f}, sigmaMAD={mad:.4f}, N={n}")

            zp_col = f"zp_{b}"
            zpe_col = f"zp_err_{b}"
            if zp_col in df.columns:
                zp = finite_array(df[zp_col])
                zpe = finite_array(df[zpe_col]) if zpe_col in df.columns else np.full(len(df), np.nan)
                zps = finite_array(df[zps_col]) if zps_col in df.columns else np.full(len(df), np.nan)
                lines.append(
                    f"  ZP                    : {np.nanmedian(zp):.5f} "
                    f"+/- {np.nanmedian(zpe):.5f}, scatter={np.nanmedian(zps):.5f}"
                )
            lines.append("")

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------
        # CMD with errors
        # ------------------------------------------------------------
        if "mag_r_AB" in df.columns and "mag_i_AB" in df.columns:
            r = finite_array(df["mag_r_AB"])
            i = finite_array(df["mag_i_AB"])
            color = r - i

            er = finite_array(df["magerr_r_AB"]) if "magerr_r_AB" in df.columns else np.full(len(df), np.nan)
            ei = finite_array(df["magerr_i_AB"]) if "magerr_i_AB" in df.columns else np.full(len(df), np.nan)
            ecolor = np.sqrt(er**2 + ei**2)

            ok = np.isfinite(color) & np.isfinite(i)
            good = (
                maybe_series(df, "good_psf_r", False).astype(bool).to_numpy()
                & maybe_series(df, "good_psf_i", False).astype(bool).to_numpy()
            )

            fig, ax = plt.subplots(figsize=(6, 7))
            ax.scatter(color[ok & ~good], i[ok & ~good], s=8, alpha=0.35, label="aper/low-quality PSF")
            ax.scatter(color[ok & good], i[ok & good], s=10, alpha=0.75, label="good PSF r+i")

            err_ok = ok & np.isfinite(ecolor) & np.isfinite(ei)

            # Plot representative error bars across the CMD, not only the brightest stars.
            idx = np.where(err_ok)[0]

            if len(idx) > 120:
                # Sort by i magnitude and sample uniformly along the sequence.
                idx_sorted = idx[np.argsort(i[idx])]
                pick = np.linspace(0, len(idx_sorted) - 1, 120).astype(int)
                idx = idx_sorted[pick]

            if len(idx):
                ax.errorbar(
                    color[idx],
                    i[idx],
                    xerr=ecolor[idx],
                    yerr=ei[idx],
                    fmt="none",
                    alpha=0.30,
                    linewidth=0.6,
                    capsize=0,
                    label="representative errors",
                )

            ax.set_xlabel("r - i [AB]")
            ax.set_ylabel("i [AB]")
            ax.set_title("Final calibrated CMD")

            ax.set_xlim(-1, 3)
            #ax.set_ylim(21, 12)
            margin = 0.5
            ax.set_ylim(np.nanmax(i[good]) + margin,
                        np.nanmin(i[good]) - margin)
            #ax.invert_yaxis()

            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------
        # SkyMapper calibration residuals from Step11
        # ------------------------------------------------------------
        sm_match_csv = reduced / "11_photcal" / "SISI_skymapper_matches.csv"

        if sm_match_csv.exists():
            sm = pd.read_csv(sm_match_csv)

            for b in bands:
                delta_col = f"delta_{b}"
                use_col = f"use_zp_{b}"
                sm_col = f"sm_{b}"

                if delta_col not in sm.columns or sm_col not in sm.columns:
                    continue

                delta = finite_array(sm[delta_col])
                smag = finite_array(sm[sm_col])

                if use_col in sm.columns:
                    used = sm[use_col].fillna(False).astype(bool).to_numpy()
                else:
                    used = np.isfinite(delta)

                ok = np.isfinite(delta) & np.isfinite(smag)

                med, mad, n = robust_stats(delta[ok & used])
                resid = delta - med

                fig, ax = plt.subplots(figsize=(6, 5))

                ax.scatter(
                    smag[ok & ~used],
                    resid[ok & ~used],
                    s=14,
                    alpha=0.35,
                    label="rejected",
                )
                ax.scatter(
                    smag[ok & used],
                    resid[ok & used],
                    s=18,
                    alpha=0.75,
                    label="used for ZP",
                )

                ax.axhline(0.0, linestyle="--", color="k")
                ax.axhline(+mad, linestyle=":", label=f"sigmaMAD={mad:.3f}")
                ax.axhline(-mad, linestyle=":")

                ax.set_ylim(-0.5, 0.5)

                ax.set_xlabel(f"SkyMapper {b} [AB]")
                ax.set_ylabel(f"SkyMapper - SISI inst. - ZP {b} [mag]")
                ax.set_title(f"Step11 SkyMapper calibration residuals: {b}")
                ax.invert_xaxis()
                ax.legend()

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # ------------------------------------------------------------
        # Per-band diagnostics
        # ------------------------------------------------------------
        for b in bands:
            mag_col = f"mag_{b}_AB"
            psf_col = f"mag_psf_{b}_AB"
            aper_col = f"mag_aper_{b}_AB"
            good_col = f"good_psf_{b}"
            qfit_col = f"psf_qfit_{b}"
            err_col = f"magerr_{b}_AB"

            if mag_col not in df.columns:
                continue

            mag = finite_array(df[mag_col])
            good = df[good_col].astype(bool).to_numpy() if good_col in df.columns else np.ones(len(df), dtype=bool)

            dmag = None
            if psf_col in df.columns and aper_col in df.columns:
                dmag = finite_array(df[psf_col]) - finite_array(df[aper_col])
                ok = np.isfinite(mag) & np.isfinite(dmag)

                # dmag vs mag.
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(mag[ok & ~good], dmag[ok & ~good], s=8, alpha=0.35, label="rejected PSF")
                ax.scatter(mag[ok & good], dmag[ok & good], s=10, alpha=0.75, label="good PSF")
                ax.axhline(0.0, linestyle="--")
                med, mad, _ = robust_stats(dmag[ok & good])
                ax.axhline(med, linestyle=":", label=f"median={med:.3f}")
                ax.axhline(med + mad, linestyle=":")
                ax.axhline(med - mad, linestyle=":")
                ax.set_xlabel(f"{b} [AB]")
                ax.set_ylabel(f"PSF - aperture {b} [mag]")
                ax.set_title(f"PSF/aperture residuals: {b}")
                ax.invert_xaxis()
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # dmag vs X.
                if "x_ref" in df.columns:
                    x = finite_array(df["x_ref"])
                    okx = ok & np.isfinite(x)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(x[okx & ~good], dmag[okx & ~good], s=8, alpha=0.30, label="rejected PSF")
                    ax.scatter(x[okx & good], dmag[okx & good], s=10, alpha=0.70, label="good PSF")
                    ax.axhline(0.0, linestyle="--")
                    add_binned_median(ax, x[okx & good], dmag[okx & good])
                    ax.set_xlabel("x_ref [pix]")
                    ax.set_ylabel(f"PSF - aperture {b} [mag]")
                    ax.set_title(f"PSF/aperture residual versus X: {b}")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                # dmag vs Y.
                if "y_ref" in df.columns:
                    y = finite_array(df["y_ref"])
                    oky = ok & np.isfinite(y)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y[oky & ~good], dmag[oky & ~good], s=8, alpha=0.30, label="rejected PSF")
                    ax.scatter(y[oky & good], dmag[oky & good], s=10, alpha=0.70, label="good PSF")
                    ax.axhline(0.0, linestyle="--")
                    add_binned_median(ax, y[oky & good], dmag[oky & good])
                    ax.set_xlabel("y_ref [pix]")
                    ax.set_ylabel(f"PSF - aperture {b} [mag]")
                    ax.set_title(f"PSF/aperture residual versus Y: {b}")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                # dmag vs color.
                if "mag_r_AB" in df.columns and "mag_i_AB" in df.columns:
                    color = finite_array(df["mag_r_AB"]) - finite_array(df["mag_i_AB"])
                    okc = ok & np.isfinite(color)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(color[okc & ~good], dmag[okc & ~good], s=8, alpha=0.30, label="rejected PSF")
                    ax.scatter(color[okc & good], dmag[okc & good], s=10, alpha=0.70, label="good PSF")
                    ax.axhline(0.0, linestyle="--")
                    add_binned_median(ax, color[okc & good], dmag[okc & good])
                    ax.set_xlabel("r - i [AB]")
                    ax.set_ylabel(f"PSF - aperture {b} [mag]")
                    ax.set_title(f"PSF/aperture residual versus color: {b}")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            # qfit versus magnitude.
            if qfit_col in df.columns:
                q = finite_array(df[qfit_col])
                ok = np.isfinite(mag) & np.isfinite(q)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(mag[ok & ~good], q[ok & ~good], s=8, alpha=0.35, label="rejected PSF")
                ax.scatter(mag[ok & good], q[ok & good], s=10, alpha=0.75, label="good PSF")
                ax.axhline(2.0, linestyle="--", label="qfit=2 cut")
                ax.set_xlabel(f"{b} [AB]")
                ax.set_ylabel("PSF qfit")
                ax.set_title(f"PSF qfit versus magnitude: {b}")
                ax.invert_xaxis()
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Magnitude error versus magnitude.
            if err_col in df.columns:
                err = finite_array(df[err_col])
                ok = np.isfinite(mag) & np.isfinite(err)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(mag[ok & ~good], err[ok & ~good], s=8, alpha=0.35, label="rejected PSF")
                ax.scatter(mag[ok & good], err[ok & good], s=10, alpha=0.75, label="good PSF")
                ax.set_xlabel(f"{b} [AB]")
                ax.set_ylabel(f"σ({b}) [mag]")
                ax.set_title(f"Photometric error versus magnitude: {b}")
                ax.set_yscale("log")
                ax.invert_xaxis()
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Source map colored by good/bad PSF.
            if "x_ref" in df.columns and "y_ref" in df.columns:
                x = finite_array(df["x_ref"])
                y = finite_array(df["y_ref"])
                ok = np.isfinite(x) & np.isfinite(y)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(x[ok & ~good], y[ok & ~good], s=8, alpha=0.35, label="rejected PSF")
                ax.scatter(x[ok & good], y[ok & good], s=10, alpha=0.75, label="good PSF")
                ax.set_xlabel("x_ref [pix]")
                ax.set_ylabel("y_ref [pix]")
                ax.set_title(f"Good/rejected PSF map: {b}")
                ax.set_aspect("equal", adjustable="box")
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"Wrote QC PDF: {pdf_path}")


if __name__ == "__main__":
    main()
