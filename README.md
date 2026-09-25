# SAMOS SISI Imaging Pipeline

Production reduction pipeline for the **SISI imaging channel of SAMOS** at SOAR.

The pipeline reduces raw SISI CCD images to astrometrically and photometrically calibrated catalogs designed to support SAMOS/DMD spectroscopy. The imaging products are therefore both imaging science products and the astrometric/photometric reference layer for the spectroscopic reduction.

## Production sequence

```text
00   Orient raw frames
01   Build superbias
02   Detector preprocessing
03   Median coaddition by band
04   Build normalized master flats
05   Apply flat fields
07c  Apply adopted linear WCS and frozen SIP distortion
09   Source detection and aperture photometry
11   SkyMapper photometric calibration
12   PSF photometry
13   Final calibrated photometric catalog
14   Merge independently reduced pointings
```

Production QC is provided for Steps 09, 11, 13, and 14.

The pipeline is configuration-driven. Instrument-wide constants and directory conventions live in `config/pipeline_config.py`; target/run-specific choices live in `config/reductions/`.

## Repository layout

```text
config/
    pipeline_config.py
    calibrations/
        SISI_T00_ADOPTED_LINEAR_WCS.fits
        SISI_SIP_SOAR_RUN8_FINAL.json
    reductions/
        sisi_Dolidze25_T00.py
        sisi_Dolidze25_T01.py

drivers/
    run_sisi_pipeline_base.py
    run_sisi_pipeline_T00.py
    run_sisi_pipeline_T01.py

pipeline/sisi/
    step00_orient/
    step01_bias/
    step02_preprocess/
    step03_coadd/
    step04_flats/
    step05_flatfield/
    step07_wcs/
    step09_photometry/
    step11_photcal/
    step12_psfphot/
    step13_final_photometry/
    step14_merge/
    utils/

qc/sisi/
    step09/
    step11/
    step13/
    step14/

data/regions/
    Dolidze25/

docs/
    TUTORIAL.md
    NEW_REDUCTION_CHECKLIST.md
    ASTROMETRY.md
```

Generated reduction products are written under `products/` and are intentionally not tracked by Git.

## Tested environment

The reference public release was tested with:

```text
Python      3.11.15
numpy       2.4.6
pandas      3.0.3
astropy     7.2.0
photutils   3.0.0
astroquery  0.4.11
matplotlib  3.10.9
```

Create the environment with:

```bash
conda env create -f environment.yml
conda activate samos-sisi
```

or install the pinned Python dependencies with:

```bash
pip install -r requirements.txt
```

Run commands from the repository root:

```bash
export PYTHONPATH=.
```

## External data layout

The pipeline code and the large observing data are kept separate. A typical project layout is:

```text
<SAMOS_PROJECT_ROOT>/
    samos-sisi-pipeline/
    _RUN8_Science_2026_01/
        SISI/
            SISI_YYYY-MM-DD/
                fits/
        SAMI/
            YYYYMMDD/
                fits/
                reduced/
```

Set the project root before running the pipeline:

```bash
export SAMOS_PROJECT_ROOT=/path/to/SAMOS/project
```

The reduction configurations derive the SISI and SAMI paths from this root. An optional external SAMOS spectroscopy-pipeline location can be supplied with `SAMOS_PIPELINE_ROOT`.

Raw FITS files are external inputs and should be treated as read-only.

## Running the reference reductions

For Dolidze 25 T00:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py \
    --from-step 00 --to-step 13 --run-qc
```

For Dolidze 25 T01:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T01.py \
    --from-step 00 --to-step 13 --run-qc
```

Run Step 14 only after both pointings have successfully reached Step 13:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py --only 14 --run-qc
```

Individual stages can be run with `--only`, for example:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py --only 05
```

For a first-time reduction of a new data set, **do not begin with a full unattended run**. Work through the stages one at a time and inspect the products after each calibration step. See [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## Astrometric calibration

The current reference reduction separates the absolute linear WCS from the instrument distortion model.

The repository contains:

```text
config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits
config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json
```

The first file preserves the adopted linear TAN WCS used for the Dolidze 25 reference reduction. The JSON file contains the frozen fifth-order SISI SIP distortion calibration.

For future routine observations, the intended operational model is that the telescope/acquisition system writes a valid preliminary linear WCS into every SISI science FITS header. That field-specific WCS is then combined with the calibrated SISI distortion model. The current public Step 00 simply flips the image data and copies the FITS header; it does **not yet transform the header WCS for that flip**. Therefore the raw-header WCS path must not be treated as production-ready until the acquisition GUI and Step 00 WCS handling are updated and validated.

The current Dolidze 25 production reduction is reproducible because Step 07c uses the bundled immutable adopted linear WCS rather than relying on the raw-header WCS.

See [`docs/ASTROMETRY.md`](docs/ASTROMETRY.md).

## Slit-region masks

The reference Dolidze 25 reductions include the validated detector-coordinate masks:

```text
data/regions/Dolidze25/
    Dolidze25-T00_0055_2026-01-13T22-38-58_pix_flipx.reg
    Dolidze25-T01_0119_2026-01-14T00-26-18_pix_flipx.reg
```

Step 09 uses these regions to flag sources falling in or near DMD slit/bar structures. The production padding is 3 pixels.

## Photometric calibration

Step 11 calibrates the instrumental photometry against SkyMapper DR4 through VizieR (`II/379/smssdr4`). Catalog queries can be cached locally under `data/skymapper/`; cached catalogs are runtime products and are not distributed with the source repository.

## Final products

For an individual pointing, the principal science product is:

```text
products/<TARGET>_<FIELD>/13_final_photometry/SISI_final_photometry.csv
```

For a two-pointing reduction, Step 14 produces:

```text
products/<TARGET>_merged_photometry/
    SISI_merged_photometry.csv
    SISI_overlap_matches_all.csv
    SISI_overlap_bar_loss.csv
```

The merged catalog preserves the independent measurements and their slit/bar provenance.

## Documentation

Start here:

- [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — first reduction from raw SISI data.
- [`docs/NEW_REDUCTION_CHECKLIST.md`](docs/NEW_REDUCTION_CHECKLIST.md) — compact operational checklist.
- [`docs/ASTROMETRY.md`](docs/ASTROMETRY.md) — linear WCS, SIP distortion, and the telescope/acquisition WCS contract.

## Reference configuration

The included Dolidze 25 T00/T01 configurations are the reproducibility reference for the current release. They should be copied and adapted for a new observing run rather than edited in place.
