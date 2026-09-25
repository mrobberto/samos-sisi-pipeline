#!/usr/bin/env python3
"""
SISI / SAMOS Imaging — Step 05: Apply Master Flats

This step flatfields the science coadds produced in Step03 using the
normalized master flats generated in Step04.

Processing consists of:

  1. Divide each science coadd by the corresponding normalized master flat.
  2. Protect against invalid (zero or non-finite) flat values.
  3. Optionally apply an empirical post-flatfield correction to the
     right detector half using configuration parameters
     (SISI_AMP_OFFSET_RIGHT and SISI_AMP_SCALE_RIGHT).

The post-flatfield correction compensates residual half-detector
photometric differences measured from the science sky background.
These corrections are dataset-dependent (e.g. T00) and are intended to
remove residual calibration-night electronics or illumination mismatches
that are not captured by the master flats alone.

Inputs
------
config.SISI_ST03_COADD
    Coadd_<band>_*.fits

config.SISI_ST04_MASTERFLATS
    MasterFlat_<band>_norm.fits

Outputs
-------
config.SISI_ST05_FLATFIELD
    Coadd_<band>_*_ff.fits

Configuration
-------------
SISI_BAND_TO_FLAT
    Mapping between science bands and master flats.

SISI_AMP_SPLIT_COL
    Column separating the left and right detector amplifiers.

SISI_AMP_OFFSET_RIGHT
    Additive correction applied to the right detector half after
    flatfielding (ADU).

SISI_AMP_SCALE_RIGHT
    Multiplicative correction applied to the right detector half after
    flatfielding.

Notes
-----
The master flats correct pixel-to-pixel sensitivity variations and
large-scale illumination structure. Any residual half-detector mismatch
remaining after flatfielding is corrected empirically using the
configuration parameters above, allowing the same pipeline to support
different observing runs and electronic configurations without changing
the reduction code.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits

import config  # expects TOP_DIR, SISI_REDUCED, etc (we only use SISI_REDUCED here)
config.build_sisi_paths()

def read_primary(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32, copy=False)
        hdr = hdul[0].header
    return data, hdr


def find_one(glob_pat: str, folder: Path) -> Path:
    matches = sorted(folder.glob(glob_pat))

    if len(matches) == 0:
        raise FileNotFoundError(f"No matches for pattern '{glob_pat}' in {folder}")

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous matches for pattern '{glob_pat}' in {folder}:\n"
            + "\n".join(str(m) for m in matches)
        )

    return matches[0]


def flatfield(science: np.ndarray, flat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if science.shape != flat.shape:
        raise ValueError(f"Shape mismatch: science {science.shape} vs flat {flat.shape}")

    good = np.isfinite(flat) & (np.abs(flat) > eps)
    out = np.full_like(science, np.nan, dtype=np.float32)
    out[good] = (science[good] / flat[good]).astype(np.float32)
    return out


def stats(a: np.ndarray) -> tuple[float, float]:
    v = a[np.isfinite(a)]
    return float(np.median(v)), float(np.std(v))

def match_science_halves(
    img,
    split_col,
    strip_width=120,
    border=80,
    gap=10,
    sigma=3.0,
):
    from astropy.stats import sigma_clip
    import numpy as np

    out = img.astype(np.float32, copy=True)

    # Inner vertical strips adjacent to the amplifier boundary.
    # Avoid:
    #   - outer image edges with zeroed pedestal pixels
    #   - a small gap around the boundary itself
    y0 = border
    y1 = img.shape[0] - border

    left_x0 = max(0, split_col - gap - strip_width)
    left_x1 = max(0, split_col - gap)

    right_x0 = min(img.shape[1], split_col + gap)
    right_x1 = min(img.shape[1], split_col + gap + strip_width)

    left = out[y0:y1, left_x0:left_x1]
    right = out[y0:y1, right_x0:right_x1]

    def sky_median(a):
        a = a[np.isfinite(a)]
        a = a[a > 0]
        if a.size == 0:
            return np.nan
        c = sigma_clip(a, sigma=sigma, maxiters=5)
        return float(np.ma.median(c))

    sky_l = sky_median(left)
    sky_r = sky_median(right)

    if np.isfinite(sky_l) and np.isfinite(sky_r) and sky_r > 0:
        scale_r = sky_l / sky_r
    else:
        scale_r = 1.0

    out[:, split_col:] *= scale_r

    return out, sky_l, sky_r, scale_r


def main():
    reduced = Path(config.SISI_REDUCED)
    coadd_dir = config.SISI_ST03_COADD
    flat_dir = config.SISI_ST04_MASTERFLATS
    out_dir = config.SISI_ST05_FLATFIELD
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    sci_r_path = find_one("Coadd_r_*.fits", coadd_dir)
    sci_i_path = find_one("Coadd_i_*.fits", coadd_dir)

    flat_r_path = flat_dir / "MasterFlat_r_norm.fits"
    flat_i_path = flat_dir / "MasterFlat_i_norm.fits"

    if not flat_r_path.exists():
        raise FileNotFoundError(f"Missing {flat_r_path}")
    if not flat_i_path.exists():
        raise FileNotFoundError(f"Missing {flat_i_path}")

    # Process bands
    for science_band, flat_band in config.SISI_BAND_TO_FLAT.items():
        sci_path = find_one(f"Coadd_{science_band}_*.fits", coadd_dir)
        flat_path = flat_dir / f"MasterFlat_{flat_band}_norm.fits"

        if not flat_path.exists():
            raise FileNotFoundError(f"Missing {flat_path}")

        sci, sci_hdr = read_primary(sci_path)
        flt, flt_hdr = read_primary(flat_path)

        med_s, std_s = stats(sci)
        med_f, std_f = stats(flt)

        ff = flatfield(sci, flt, eps=1e-6)

        amp_offset_right = getattr(config, "SISI_AMP_OFFSET_RIGHT", {})
        amp_scale_right = getattr(config, "SISI_AMP_SCALE_RIGHT", {})

        amp_offset_r = float(amp_offset_right.get(science_band, 0.0))
        amp_scale_r = float(amp_scale_right.get(science_band, 1.0))

        # Apply additive pedestal correction first, then multiplicative gain correction.
        # The offset is in the same units as the flatfielded science image.
        right = ff[:, config.SISI_AMP_SPLIT_COL:]
        ff[:, config.SISI_AMP_SPLIT_COL:] = (right - amp_offset_r) * amp_scale_r


        med_ff, std_ff = stats(ff)

        out_path = out_dir / f"{sci_path.stem}_ff.fits"

        hdr = sci_hdr.copy()
        hdr["HISTORY"] = f"Flatfielded: {sci_path.name} / {flat_path.name}"
        hdr["FLATFILE"] = (flat_path.name, "Master flat used")
        hdr["FLATNORM"] = (True, "Flat expected normalized ~1")
        hdr["BAND"] = (science_band, "Science band")
        hdr["FLATBAND"] = (flat_band, "Master flat band")

        hdr["QC_S_MED"] = (med_s, "Median science pre-FF")
        hdr["QC_S_STD"] = (std_s, "Std science pre-FF")
        hdr["QC_F_MED"] = (med_f, "Median flat")
        hdr["QC_F_STD"] = (std_f, "Std flat")
        hdr["QC_FF_MD"] = (med_ff, "Median science post-FF")
        hdr["QC_FF_ST"] = (std_ff, "Std science post-FF")
        hdr["AMPCORR"] = (True, "Post-flatfield right-half gain correction")
        hdr["AMPOFFR"] = (amp_offset_r, "Post-flatfield right-half offset")
        hdr["AMPSCLR"] = (amp_scale_r, "Post-flatfield right-half scale")

        fits.writeto(out_path, ff.astype(np.float32), hdr, overwrite=True)

        print(f"\nBand {science_band} using flat {flat_band}")
        print("  science:", sci_path.name, f"median={med_s:.3f} std={std_s:.3f}")
        print("  flat   :", flat_path.name, f"median={med_f:.6f} std={std_f:.6f}")
        print("  ff out :", out_path.name, f"median={med_ff:.3f} std={std_ff:.3f}")

    print("\nDone. Output folder:", out_dir)


if __name__ == "__main__":
    main()
