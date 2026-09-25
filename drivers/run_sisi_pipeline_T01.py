#!/usr/bin/env python3
"""
run
===
$ PYTHONPATH=. python drivers/run_sisi_pipeline_T01.py --from-step 01 --to-step 13 --run-qc
"""

import os

os.environ["SISI_ACTIVE_REDUCTION"] = "config.reductions.sisi_Dolidze25_T01"

from run_sisi_pipeline_base import main

STAGES = {
    "00": "pipeline/sisi/step00_orient/step00_orient_raw.py",
    "01": "pipeline/sisi/step01_bias/step01_make_superbias.py",
    "02": "pipeline/sisi/step02_preprocess/step02_detector_preprocess.py",
    "03": "pipeline/sisi/step03_coadd/step03_median_coadd_bands.py",
    "04": "pipeline/sisi/step04_flats/step04_make_masterflats.py",
    "05": "pipeline/sisi/step05_flatfield/step05_apply_flats.py",
    "07c": "pipeline/sisi/step07_wcs/step07c_copy_wcs_to_science.py",
    "09": "pipeline/sisi/step09_photometry/step09_detect_photometry.py",
    "11": "pipeline/sisi/step11_photcal/step11_skymapper_calibrate.py",
    "12": "pipeline/sisi/step12_psfphot/step12_psf_photometry.py",
    "13": "pipeline/sisi/step13_final_photometry/step13_final_photometry.py",
    "14": "pipeline/sisi/step14_merge/step14_merge_T00_T01.py",
}

QC_STAGES = {
    "09": "qc/sisi/step09/qc_step09_detect_photometry.py",
    "11": "qc/sisi/step11/qc_step11_skymapper_photcal.py",
    "13": "qc/sisi/step13/qc_step13_final_photometry.py",
    "14": "qc/sisi/step14/qc_step14_merged_photometry.py",
}

if __name__ == "__main__":
    main(STAGES, QC_STAGES)