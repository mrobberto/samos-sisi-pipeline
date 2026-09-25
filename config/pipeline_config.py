from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

GAIN_E_PER_ADU = 2.1
READNOISE_E = 3.8

DATA_DIR = PIPELINE_ROOT / "data"
REGION_DATA_DIR = DATA_DIR / "regions"
#
CALIB_DIR = PIPELINE_ROOT / "config" / "calibrations"
SISI_SIP_CALIBRATION = ( CALIB_DIR / "SISI_SIP_SOAR_RUN8_FINAL.json" )

"""PRODUCTS_DIR, but not QC"""
SISI_DIR_01_BIAS = "01_bias"
SISI_DIR_02_PREPROCESS = "02_preprocess"
SISI_DIR_02_PREPROCESS_FLATS = "02_preprocess_flats"
SISI_DIR_03_COADD = "03_coadd"
SISI_DIR_04_FLATS = "04_flats"
SISI_DIR_04_FLAT_PREPROC = "04_flats/flat_preproc"
SISI_DIR_04_MASTERFLATS = "04_flats/masterflats"
SISI_DIR_05_FLATFIELD = "05_flatfield"
SISI_DIR_06_ORIENTATION = "06_orientation"
SISI_DIR_07_WCS = "07_wcs"
#SISI_DIR_08_SLIT_MAPPING = "08_slit_mapping"
SISI_DIR_09_PHOTOMETRY = "09_photometry"
SISI_DIR_10_ASTROMETRY = "10_astrometry"
SISI_DIR_11_PHOTCAL = "11_photcal"
SISI_DIR_11_PHOTCAL_QC = "11_photcal_qc"
SISI_DIR_12_PSFPHOT = "12_psfphot"
SISI_DIR_12_PSFPHOT_QC = "12_psfphot_qc"
SISI_DIR_13_FINAL_PHOTOMETRY = "13_final_photometry"
SISI_DIR_13_FINAL_PHOTOMETRY_QC = "13_final_photometry_qc"

def print_config():
    """Print the currently active reduction configuration."""
    import config

    print("Active SISI reduction:")
    print(f"  TARGET       = {config.SISI_TARGET}")
    print(f"  FIELD        = {config.SISI_FIELD}")
    print(f"  SISI_NIGHT   = {config.SISI_NIGHT}")
    print(f"  TOP_DIR      = {config.TOP_DIR}")
    print(f"  SISI_ROOT    = {config.SISI_ROOT}")
    print(f"  SISI_REDUCED = {config.SISI_REDUCED}")