#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC plots for SISI Step 09 blind-detection photometry."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config


def finite_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    ok = np.ones(len(df), dtype=bool)
    for c in cols:
        ok &= np.isfinite(df[c].to_numpy())
    return df.loc[ok].copy()


def scatter_cmd(ax, df, color_col, mag_col, title):
    d = finite_df(df, [color_col, mag_col])
    ax.scatter(d[color_col], d[mag_col], s=8, alpha=0.55)
    ax.invert_yaxis()
    ax.set_xlabel(color_col.replace("color_", "").replace("_", " - "))
    ax.set_ylabel(mag_col.replace("mag_inst_", "") + " instrumental mag")
    ax.set_title(f"{title}  N={len(d)}")
    ax.grid(alpha=0.25)


def mag_to_size(mag):
    m = np.asarray(mag, dtype=float)
    ok = np.isfinite(m)
    s = np.full(len(m), 8.0)
    if ok.sum() > 0:
        lo, hi = np.nanpercentile(m[ok], [5, 95])
        denom = max(hi - lo, 1e-6)
        # brighter = smaller mag = larger marker
        val = 1.0 - np.clip((m - lo) / denom, 0, 1)
        s = 8 + 60 * val
    return s


def main():
    parser = argparse.ArgumentParser(description="QC plots for SISI detected-source photometry.")
    parser.add_argument("--input", default=None, help="Photometry CSV. Default: Step09 field catalog.")
    args = parser.parse_args()

    phot_dir = Path(config.SISI_REDUCED) / "09_photometry"
    qc_dir = Path(config.SISI_REDUCED) / "09_photometry_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    infile = Path(args.input) if args.input else phot_dir / "SISI_detected_field_photometry.csv"
    if not infile.exists():
        raise FileNotFoundError(f"Missing photometry catalog: {infile}")

    df = pd.read_csv(infile)
    print(f"Loaded {len(df)} sources from {infile}")

    pdf_path = qc_dir / "qc_step09_detect_photometry.pdf"
    map_path = qc_dir / "qc_step09_source_map_color_ri_size_r.png"
    cmd_ri_path = qc_dir / "qc_step09_cmd_r_vs_ri.png"

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        ax.axis("off")
        lines = [
            "SISI Step 09 detected-source photometry QC",
            f"Catalog: {infile.name}",
            f"N sources: {len(df)}",
        ]
        for c in ["mag_inst_r", "mag_inst_i", "mag_inst_z", "color_r_i", "color_i_z"]:
            if c in df.columns:
                lines.append(f"finite {c}: {np.isfinite(df[c]).sum()}")
        ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # CMDs
        cmd_specs = [
            ("color_r_i", "mag_inst_r", "r vs r-i"),
            ("color_r_i", "mag_inst_i", "i vs r-i"),
            ("color_i_z", "mag_inst_i", "i vs i-z"),
            ("color_r_z", "mag_inst_r", "r vs r-z"),
        ]
        first_cmd_saved = False
        for color_col, mag_col, title in cmd_specs:
            if color_col not in df.columns or mag_col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 6.2))
            scatter_cmd(ax, df, color_col, mag_col, title)
            pdf.savefig(fig, bbox_inches="tight")
            if not first_cmd_saved and color_col == "color_r_i" and mag_col == "mag_inst_r":
                fig.savefig(cmd_ri_path, dpi=160, bbox_inches="tight")
                first_cmd_saved = True
            plt.close(fig)

        # Source map: color by r-i, size by r brightness.

        # Support both the old (x,y) and the new (x_detect_<band>, y_detect_<band>) catalogs.
        if {"x", "y"}.issubset(df.columns):
            xcol, ycol = "x", "y"
        else:
            xcols = [c for c in df.columns if c.startswith("x_detect_")]
            ycols = [c for c in df.columns if c.startswith("y_detect_")]

            if len(xcols) == 1 and len(ycols) == 1:
                xcol, ycol = xcols[0], ycols[0]
            else:
                xcol = ycol = None

        if xcol is not None:
            fig, ax = plt.subplots(figsize=(7.0, 7.0))

            if "mag_inst_r" in df.columns:
                sizes = mag_to_size(df["mag_inst_r"].to_numpy())
            else:
                sizes = np.full(len(df), 12.0)

            if "color_r_i" in df.columns:
                color = df["color_r_i"].to_numpy()
                sc = ax.scatter(df[xcol], df[ycol],
                                c=color, s=sizes, alpha=0.65)
                cb = fig.colorbar(sc, ax=ax)
                cb.set_label("r - i instrumental color")
            else:
                ax.scatter(df[xcol], df[ycol],
                           s=sizes, alpha=0.65)

            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")
            ax.set_title("Detected field sources: color = r-i, size = r brightness")
            ax.grid(alpha=0.2)

            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(map_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


        # Photometric scatter/error diagnostic if available.
        if {"mag_inst_r", "mag_err_r"}.issubset(df.columns):
            d = finite_df(df, ["mag_inst_r", "mag_err_r"])
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            ax.scatter(d["mag_inst_r"], d["mag_err_r"], s=8, alpha=0.55)
            ax.set_xlabel("r instrumental mag")
            ax.set_ylabel("r mag error estimate")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(f"Photometric uncertainty diagnostic  N={len(d)}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Wrote:")
    print(f"  {pdf_path}")
    if map_path.exists():
        print(f"  {map_path}")
    if cmd_ri_path.exists():
        print(f"  {cmd_ri_path}")


if __name__ == "__main__":
    main()
