import os
from pathlib import Path

from config.pipeline_config import REGION_DATA_DIR, PIPELINE_ROOT

# -----------------------------------------------------------------------------
# Project / data roots
# -----------------------------------------------------------------------------

# By default, assume the repository lives inside the SAMOS project directory:
#
#   <SAMOS_PROJECT_ROOT>/
#       sisi_pipeline/
#       _RUN8_Science_2026_01/
#
# The root can be overridden for another installation with:
#
#   export SAMOS_PROJECT_ROOT=/path/to/SAMOS/project
#
SAMOS_DIR = Path(
    os.environ.get("SAMOS_PROJECT_ROOT", str(PIPELINE_ROOT.parent))
).expanduser().resolve()

# Repository-local files
SISI_TOP_DIR = PIPELINE_ROOT

# External observing data
TOP_DIR = SAMOS_DIR / "_RUN8_Science_2026_01"


SISI_TARGET = "Dolidze25"
SISI_FIELD = "T01"        # <=== CHECK THIS
SISI_NIGHT = "SISI_2026-01-13"
SAMI_NIGHT_ID = "20260113"

SAMI_TOP_DIR = Path(
    os.environ.get(
        "SAMOS_PIPELINE_ROOT",
        str(SAMOS_DIR / "samos-pipeline"),
    )
).expanduser().resolve()

SAMI_NIGHT_DIR = TOP_DIR / "SAMI" / SAMI_NIGHT_ID
# -----------------------------------------------------------------------------
# Compatibility aliases for scripts that still use spectroscopy names
RUN_ROOT = SAMI_NIGHT_DIR
REDUCED = SAMI_NIGHT_DIR / "reduced"
FITS = SAMI_NIGHT_DIR / "fits"

# -----------------------------------------------------------------------------
# External SAMI spectroscopy products used for SISI/SAMOS cross-registration
# -----------------------------------------------------------------------------
SAMI_REDUCED = TOP_DIR / "SAMI" / SISI_TARGET / "reduced"
SAMI_ST04_PIXFLAT = SAMI_REDUCED / "04_traces"
ST04_PIXFLAT = SAMI_ST04_PIXFLAT

SAMI_EVEN_TRACE_MASK = SAMI_ST04_PIXFLAT / "Even_traces_mask.fits"
SAMI_EVEN_TRACE_SLITID = SAMI_ST04_PIXFLAT / "Even_traces_slitid.fits"
# -----------------------------------------------------------------------------
# Backward-compatible alias used by current scripts
ST04_PIXFLAT = SAMI_ST04_PIXFLAT
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

SISI_ROOT = TOP_DIR / "SISI" / SISI_NIGHT
SISI_PRODUCT_ROOT = PIPELINE_ROOT / "products" / f"{SISI_TARGET}_{SISI_FIELD}"
SISI_REDUCED = SISI_PRODUCT_ROOT
SISI_RAW = SISI_ROOT / "fits"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
SISI_ST00_FLIPPED = SISI_REDUCED / "00_orient"
SISI_ST00_ORIENT = SISI_ST00_FLIPPED

SISI_ST01_BIAS = SISI_REDUCED / "01_bias"

SISI_ST02_PREPROCESS = SISI_REDUCED / "02_preprocess"
SISI_DIR_02_PREPROCESS = SISI_ST02_PREPROCESS

SISI_ST02_PREPROCESS_FLATS = SISI_REDUCED / "02_preprocess_flats"
SISI_DIR_02_PREPROCESS_FLATS = SISI_ST02_PREPROCESS_FLATS

SISI_ST03_COADD = SISI_REDUCED / "03_coadd"

SISI_ST04_FLAT_PREPROC = SISI_REDUCED / "04_flats" / "flat_preproc"
SISI_ST04_MASTERFLATS = SISI_REDUCED / "04_flats" / "masterflats"

SISI_ST05_FLATFIELD = SISI_REDUCED / "05_flatfield"
SISI_ST06_ORIENT = SISI_REDUCED / "06_orientation"
SISI_ST07_WCS = SISI_REDUCED / "07_wcs"


SISI_ST09_PHOTOMETRY = SISI_REDUCED / "09_photometry"
SISI_ST10_ASTROMETRY = SISI_REDUCED / "10_astrometry"

SISI_ST11_PHOTCAL = SISI_REDUCED / "11_photcal"
SISI_ST11_PHOTCAL_QC = SISI_REDUCED / "11_photcal_qc"

SISI_ST12_PSFPHOT = SISI_REDUCED / "12_psfphot"
SISI_ST12_PSFPHOT_QC = SISI_REDUCED / "12_psfphot_qc"

SISI_ST13_FINAL_PHOTOMETRY = SISI_REDUCED / "13_final_photometry"
SISI_ST13_FINAL_PHOTOMETRY_QC = SISI_REDUCED / "13_final_photometry_qc"

#----------------------------------------------------------------------------
# Instrument / astrometric constants for this reduction
# -----------------------------------------------------------------------------
SISI_AMP_SPLIT_COL = 528
SISI_PIXSCALE_ARCSEC = 0.184

SISI_RA0_DEG = 101.269629
SISI_DEC0_DEG = 0.2297941

# -----------------------------------------------------------------------------
# Frame selections for this target/run
# -----------------------------------------------------------------------------
SISI_FRAME_PATTERNS = {
    "bias": "{num:03d}.bias_*.fits",
    "science": "{num:03d}.sci_sloan-{band}_{target}-{field}.fits",
    "flat": "{num:03d}.flat_sloan-{flat_band}_Flat.fits",
}
# -----------------------------------------------------------------------------
# CHECK FOR UNUSUAL/WRONG DATA FORMAT; FIX implememts in step02
SISI_UINT16_MODE = "pedestal_aware"  #For Dolidze 25-T00, with bad electronics setup
#SISI_UINT16_MODE = "negative_only"   #For Dolidze 25-T01
# -----------------------------------------------------------------------------

SISI_BIAS_FRAMES = range(4, 24)

SISI_SCIENCE_FRAMES = {
    "r": range(119, 124), #73:77 RANGE EXCLUDES THE LAST NUMBER, i.e  range(0,3) means 0, 1, 2.
    "i": range(124, 134), #78:82
#    "i1": range(83, 94), #83:93 BAD CCD SETUP
    "z": range(134, 139), #94:104
}

SISI_EXPTIME_R = 500.0
SISI_EXPTIME_I = 500.0
SISI_EXPTIME_Z = 1000.0

# -----------------------------------------------------------------------------
# GAIN CORRECTION TO BE APPLIED TO THE RIGHT SIDE
# -----------------------------------------------------------------------------
SISI_AMP_OFFSET_RIGHT = {
    "r": 0.0,
    "i": 0.0,
    "z": 0.0,
}
SISI_AMP_SCALE_RIGHT = {
    "r": 1.2542,#,0.72,
    "i": 1.1701,#,0.6999906456563578,
    "z": 1.4136,#0.695,
}

# ------------------------------------------------------------------
# STEP02 PROCESSING
# ------------------------------------------------------------------

# Repair pedestal-aware int16/u16 encoding on science frames
SISI_STEP02_SCIENCE_PED_U16_FIX = True

# Fields for which the repair applies
SISI_STEP02_SCIENCE_PED_U16_FIELDS = ("T01",)

# Per-amplifier pedestals
SISI_STEP02_SCIENCE_PED_LEFT  = 518.0
SISI_STEP02_SCIENCE_PED_RIGHT = 533.0

# Recover wrapped int16 values in flats
SISI_STEP02_FLAT_U16_FIX = True

# Recover wrapped values in other science frames
SISI_STEP02_OTHER_SCIENCE_U16_MODE = "none"

# Future gain equalization
SISI_STEP02_AMP_MATCH = False

# Miscellaneous
SISI_STEP02_OUTPUT_SUFFIX = "_ampmatch"
SISI_STEP02_WRITE_DEBUG = False

# ------------------------------------------------------------------
# STEP04 PROCESSING
# ------------------------------------------------------------------

SISI_STEP04_DO_AMP_EQUALIZE = False
SISI_STEP04_DO_ILLUM_MATCH_HALVES = False
SISI_STEP04_DO_SUPERBIAS_SUBTRACT = False
SISI_STEP04_USE_I_DIFFERENTIAL_FLAT = False
SISI_STEP04_MIN_FLATS_PER_GROUP = 3

# -----------------------------------------------------------------------------
# FLAT FRAMES MAY HAVE BEEN TAKEN AT ANITER TIME....
# -----------------------------------------------------------------------------
SISI_FLAT_NIGHT = "SISI_2026-01-16"
SISI_FLAT_ROOT = TOP_DIR / "SISI" / SISI_FLAT_NIGHT

SISI_FLAT_FRAMES = {
    "r": range(15, 25),
    "i_025s": range(65, 75),
    "i_05s": range(75, 85),
    "i_1s": range(55, 65),
    "z": range(85, 95), #85, 95),
}

SISI_FLAT_MASTER_BANDS = {
    "r": ["r"],
    "i": ["i_05s"],
    "z": ["z"],
}

SISI_BAND_TO_FLAT = {
    "r": "r",
    "i": "i",
#    "i1": "i",
    "z": "z",
}

# -----------------------------------------------------------------------------
# Cached catalogs for this field
# -----------------------------------------------------------------------------
SISI_GAIA_CACHE_DIR = SISI_TOP_DIR / "data" / "gaia" / SISI_TARGET
SISI_GAIA_CACHE_CSV = SISI_GAIA_CACHE_DIR / f"{SISI_TARGET}_{SISI_FIELD}_gaia_dr3.csv"
SISI_SKYMAPPER_CACHE_DIR = SISI_TOP_DIR / "data" / "skymapper" / SISI_TARGET
SISI_SKYMAPPER_CACHE_CSV = SISI_SKYMAPPER_CACHE_DIR / f"{SISI_TARGET}_{SISI_FIELD}_skymapper.csv"
#SISI_SKYMAPPER_CACHE_CSV = SISI_TOP_DIR / "data" / "skymapper" / SISI_TARGET / "SISI_skymapper_query.csv"

# -----------------------------------------------------------------------------
# Target-specific ancillary files kept inside sisi_pipeline/data/
# -----------------------------------------------------------------------------
TARGET_REGION_DIR = REGION_DATA_DIR / SISI_TARGET

SISI_PIXEL_REGION_BASENAME = "Dolidze25-T01_0119_2026-01-14T00-26-18_pix_flipx.reg"
SISI_RADEC_CSV_BASENAME = "Dolidze25-T01Even_RADEC=101.269629+0.2297941.csv"

SISI_PIXEL_REGION = TARGET_REGION_DIR / SISI_PIXEL_REGION_BASENAME
SISI_RADEC_CSV = TARGET_REGION_DIR / SISI_RADEC_CSV_BASENAME

# -----------------------------------------------------------------------------
# ASTROMETRY HEADER
# -----------------------------------------------------------------------------
SISI_WCS_REF_FINAL = (
    PIPELINE_ROOT
    / "config"
    / "calibrations"
    / "SISI_T00_ADOPTED_LINEAR_WCS.fits"
)
SISI_WCS_REF_FRAME = SISI_WCS_REF_FINAL

SISI_SIP_CALIBRATION = (
    PIPELINE_ROOT
    / "config"
    / "calibrations"
    / "SISI_SIP_SOAR_RUN8_FINAL.json"
)
# -----------------------------------------------------------------------------
# Special reference files for this target/run
# -----------------------------------------------------------------------------
SISI_QUARTZ_FLIPX_BASENAME = "203.sci_sloan-r_D25Quartz_flipx.fits"
SISI_WCS_REF_BASENAME = "203_sci_sloan-r_D25Quartz_wcs.fits"

SISI_QUARTZ_FLIPX_REF = (
    SISI_REDUCED / "06_orientation" / "203.sci_sloan-r_D25Quartz_orient.fits"
)

SISI_PIXEL_REGION = (
    SISI_TOP_DIR /
    "data/regions/Dolidze25/" /
    SISI_PIXEL_REGION_BASENAME
)

SISI_RADEC_CSV = (
    SISI_TOP_DIR /
    "data/regions/Dolidze25/" /
    SISI_RADEC_CSV_BASENAME
)

# -----------------------------------------------------------------------------
# PSF FITTING PARAMETERS
# -----------------------------------------------------------------------------

SISI_PSF_MODEL = "moffat"
SISI_PSF_FWHM_FORCE = 2.5
SISI_MOFFAT_BETA = 2.0