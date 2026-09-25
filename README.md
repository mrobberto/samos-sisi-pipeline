# SAMOS SISI Imaging Pipeline

Production imaging-reduction and photometric-analysis pipeline for the SISI
imaging channel of SAMOS.

This repository contains the code used for the Dolidze 25 T00/T01 reductions,
including the frozen astrometric calibration and the detector-coordinate
region masks required by the production photometry.

Raw observing data and generated reduction products are not distributed here.

## Production sequence

```text
00   Orient raw frames
01   Build superbias
02   Detector preprocessing
03   Median coaddition
04   Build master flats
05   Apply flat fields
07c  Apply adopted linear WCS and frozen SIP distortion
09   Source detection and aperture photometry
11   SkyMapper DR4 photometric calibration
12   Fixed-position PSF photometry
13   Final calibrated catalog
14   Merge T00 and T01 catalogs

```

Production QC is provided for Steps 09, 11, 13, and 14.

## Tested environment

Python 3.11.15 with package versions recorded in `environment.yml` and `requirements.txt`.

## Running

From the repository root:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py --from-step 00 --to-step 13 --run-qc
PYTHONPATH=. python drivers/run_sisi_pipeline_T01.py --from-step 00 --to-step 13 --run-qc
```

Step 14 should be run after both fields have completed through Step 13.

## External data

By default, the reduction assumes:

```text
<SAMOS_PROJECT_ROOT>/
    samos-sisi-pipeline-public/
    _RUN8_Science_2026_01/
        SISI/
        SAMI/
```

Set another location with:

```bash
export SAMOS_PROJECT_ROOT=/path/to/SAMOS/project
```

## Astrometric calibration

The production astrometric calibration consists of:

```text
config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits
config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json
```

The FITS file carries the adopted linear TAN WCS and detector geometry.
The JSON file carries the frozen fifth-order SIP distortion solution.

## Slit-region masks

Step 09 uses the following detector-coordinate DS9 region files:

```text
data/regions/Dolidze25/Dolidze25-T00_0055_2026-01-13T22-38-58_pix_flipx.reg
data/regions/Dolidze25/Dolidze25-T01_0119_2026-01-14T00-26-18_pix_flipx.reg
```

The production slit-mask padding is 3 pixels.
