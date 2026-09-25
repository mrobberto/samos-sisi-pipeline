# First Reduction with the SAMOS SISI Imaging Pipeline

This tutorial is written for a user reducing SISI data for the first time. It starts with raw SISI FITS images and ends with a calibrated photometric catalog suitable for use by the SAMOS spectroscopic reduction.

The most important rule is simple:

> **For the first reduction of a new observing run, execute and inspect the pipeline one stage at a time.**

Do not use a full unattended `00 -> 13` run until one field has been reduced successfully and the detector, flat-field, astrometric, and photometric behavior have been checked.

## 1. What the pipeline produces

SISI is the imaging channel of SAMOS. Its reduction pipeline does more than generate calibrated images: it produces the astrometric and photometric reference catalog needed to identify objects, relate celestial coordinates to the DMD/slit geometry, and connect imaging measurements with the spectroscopic reduction.

The production sequence is:

```text
raw SISI FITS
    |
    v
00  orientation
01  superbias
02  detector preprocessing
03  coaddition
04  master-flat construction
05  flat-fielding
07c astrometric WCS + SIP distortion
09  source detection + aperture photometry
11  SkyMapper calibration
12  PSF photometry
13  final calibrated catalog
14  optional multi-pointing merge
```

## 2. Install the software

Clone the repository and enter it:

```bash
git clone <repository-url>
cd samos-sisi-pipeline
```

Create the tested environment:

```bash
conda env create -f environment.yml
conda activate samos-sisi
```

Run all commands below from the repository root:

```bash
export PYTHONPATH=.
```

## 3. Organize the observing data

The pipeline code is kept separate from the large observing data. A normal layout is:

```text
<SAMOS_PROJECT_ROOT>/
    samos-sisi-pipeline/
    _RUN8_Science_YYYY_MM/
        SISI/
            SISI_YYYY-MM-DD/
                fits/
        SAMI/
            YYYYMMDD/
                fits/
                reduced/
```

Set the project root:

```bash
export SAMOS_PROJECT_ROOT=/path/to/SAMOS/project
```

The original raw FITS directories should be treated as read-only. Pipeline products are written beneath the repository `products/` directory.

## 4. Inventory the observing night before editing the configuration

Before running any script, identify:

- science-night bias frames;
- science exposures belonging to each target/pointing;
- filter used by each sequence;
- exposure times;
- flat-field night;
- flat sequences and exposure times;
- approximate target coordinates;
- the relationship between the SISI exposure and the corresponding SAMI/SAMOS observation.

The pipeline expects numbered files and the reference configurations use patterns such as:

```text
NNN.bias_*.fits
NNN.sci_sloan-<band>_<target>-<field>.fits
NNN.flat_sloan-<band>_Flat.fits
```

The reduction configuration explicitly selects the accepted frames. For example:

```python
SISI_BIAS_FRAMES = range(4, 24)

SISI_SCIENCE_FRAMES = {
    "r": range(73, 78),
    "i": range(78, 83),
}
```

Remember that Python excludes the upper end of `range`: `range(73, 78)` means 73, 74, 75, 76, and 77.

Flats may come from a different night:

```python
SISI_FLAT_NIGHT = "SISI_YYYY-MM-DD"

SISI_FLAT_FRAMES = {
    "r": range(...),
    "i_025s": range(...),
    "i_05s": range(...),
    "z": range(...),
}
```

## 5. Create a reduction configuration

Use one of the validated Dolidze 25 files as a template:

```bash
cp config/reductions/sisi_Dolidze25_T00.py \
   config/reductions/sisi_MyTarget_T00.py
```

Edit the new file, not the reference configuration.

At minimum review:

```python
SISI_TARGET
SISI_FIELD
SISI_NIGHT
SAMI_NIGHT_ID

SISI_RA0_DEG
SISI_DEC0_DEG

SISI_BIAS_FRAMES
SISI_SCIENCE_FRAMES

SISI_EXPTIME_R
SISI_EXPTIME_I
SISI_EXPTIME_Z

SISI_FLAT_NIGHT
SISI_FLAT_FRAMES
SISI_FLAT_MASTER_BANDS
SISI_BAND_TO_FLAT
```

Also review all Step 02 detector-recovery settings and the post-flat right-amplifier corrections. These are observing-run-dependent and must not be copied blindly.

## 6. Create a driver for the reduction

Copy the closest reference driver:

```bash
cp drivers/run_sisi_pipeline_T00.py \
   drivers/run_sisi_pipeline_MyTarget_T00.py
```

Change only the active reduction import:

```python
os.environ["SISI_ACTIVE_REDUCTION"] = \
    "config.reductions.sisi_MyTarget_T00"
```

The production stage sequence should normally remain unchanged.

The driver supports:

```text
--from-step
--to-step
--only
--run-qc
```

For example:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 00
```

## 7. Detector preprocessing: understand the data before choosing corrections

SISI commissioning data exposed an important distinction between science frames and flat fields.

Science data can require recovery of signed-int16 values and, for particular electronics configurations, pedestal-aware reconstruction. Flat fields must preserve the detector response needed to build the master flat. They therefore use only the generic integer-format recovery appropriate to the raw data and should not inherit science-frame pedestal subtraction or amplifier equalization unless a separate flat-specific calibration has been established.

Important configuration controls include:

```python
SISI_STEP02_SCIENCE_PED_U16_FIX
SISI_STEP02_SCIENCE_PED_U16_FIELDS
SISI_STEP02_SCIENCE_PED_LEFT
SISI_STEP02_SCIENCE_PED_RIGHT

SISI_STEP02_FLAT_U16_FIX
SISI_STEP02_OTHER_SCIENCE_U16_MODE
SISI_STEP02_AMP_MATCH
SISI_STEP02_WRITE_DEBUG
```

For a new run, inspect raw pixel statistics first. Do not assume the Run 8 electronics behavior applies unchanged.

Short flat exposures also require care. If the shutter behaves as an iris, the center of the detector can receive a longer effective exposure than the edges during very short integrations. A center-bright pattern in short flats is therefore not automatically a sensitivity gradient.

## 8. Step 00 — orient the configured raw frames

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 00
```

Step 00 reads the configured bias, science, and flat exposures and applies the SISI left-right orientation correction using `np.fliplr`.

After Step 00, stop and inspect representative images.

Check:

- expected dimensions;
- correct orientation;
- no missing configured frames;
- no zero-byte or corrupt FITS files;
- saturation behavior;
- bad columns and obvious detector defects;
- consistency between science and flat frames.

### Important WCS note

The current Step 00 flips the image data and copies the FITS header. It does **not yet transform an input WCS to account for the horizontal flip**. Therefore a telescope/acquisition WCS present in the raw file must not yet be assumed to remain geometrically correct after Step 00.

This is a known interface item for the acquisition GUI/pipeline: future routine SISI raw files should always contain a valid preliminary WCS, and Step 00 must propagate that WCS through the orientation transform. Until that behavior is implemented and validated, the reference Run 8 reduction uses the frozen adopted WCS applied later in Step 07c.

See `docs/ASTROMETRY.md`.

## 9. Step 01 — build the superbias

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 01
```

Products are written under:

```text
products/<TARGET>_<FIELD>/01_bias/
```

Inspect the superbias before proceeding. Verify the overall level, both amplifier halves, persistent detector structure, and any unexpected gradients or discontinuities.

## 10. Step 02 — detector preprocessing

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 02
```

This stage applies the active reduction's detector-format recovery and science-frame pedestal handling.

For science frames inspect:

- left/right background levels;
- pixel minimum and maximum values;
- stars near the amplifier boundary;
- saturated sources;
- whether recovered data remain physically plausible.

For flats inspect:

- large-scale illumination remains present;
- detector-response differences have not been artificially removed;
- no science pedestal correction has leaked into the flat processing.

Do not proceed by trying to "fix" a Step 02 problem with the flat field.

## 11. Step 03 — median coaddition by band

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 03
```

Products are written under:

```text
products/<TARGET>_<FIELD>/03_coadd/
```

Inspect each coadd for the expected input frames, sharp stellar profiles, sensible outlier rejection, preservation of dynamic range, and any amplifier-half discontinuity.

## 12. Step 04 — build normalized master flats

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 04
```

Products include:

```text
products/<TARGET>_<FIELD>/04_flats/
    MasterFlat_<band>_norm.fits
```

The master flat is intended to preserve and measure multiplicative detector response. Production defaults therefore avoid using amplifier equalization or illumination matching as a substitute for the measured flat response.

Inspect every master flat for finite values, pixel-scale sensitivity structure, amplifier-boundary behavior, dust features, large-scale gradients, and shutter-pattern contamination.

## 13. Step 05 — apply the flats

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 05
```

Products are written under:

```text
products/<TARGET>_<FIELD>/05_flatfield/
```

Each science coadd is divided by its configured normalized master flat. The stage can then apply a reduction-specific residual correction to the right detector half:

```python
SISI_AMP_OFFSET_RIGHT
SISI_AMP_SCALE_RIGHT
```

These values are empirical run/field calibration parameters, not universal detector constants.

Inspect the flat-fielded sky on both amplifier halves. A residual boundary should be understood before moving to astrometry and photometry.

## 14. Step 07c — apply the astrometric reference and SIP model

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 07c
```

The current production Step 07c does **not** solve a new astrometric model. It copies an adopted linear TAN WCS from an immutable reference FITS file and attaches the frozen SISI SIP distortion model.

The reference release contains:

```text
config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits
config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json
```

For Dolidze 25, these provide a reproducible frozen astrometric state.

For future observations, the intended operational model is:

```text
telescope acquisition / SAMOS GUI
        |
        v
raw SISI FITS with valid preliminary linear WCS
        |
        v
Step 00 orientation with WCS transformed consistently
        |
        v
field-specific linear WCS
        +
frozen instrument SIP distortion
        |
        v
final TAN-SIP WCS
```

That path is an operational requirement, not yet the behavior of the current Step 00. Until the GUI and Step 00 are updated and validated, a new reduction must provide an independently validated immutable linear WCS reference to Step 07c.

After Step 07c, inspect the resulting WCS against external stars. A global pointing error is a linear-WCS problem; it should not be absorbed by arbitrarily changing the SIP distortion.

## 15. Step 09 — source detection and aperture photometry

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py \
    --only 09 --run-qc
```

The production implementation selects one reference detection band, with preference `i`, then `r`, then `z`. Detected positions are projected consistently into the other bands for multi-band photometry.

Typical products are written under:

```text
products/<TARGET>_<FIELD>/09_photometry/
```

Step 09 also evaluates a configured DS9 pixel-region mask and flags sources that lie in or near DMD slit/bar structures. The reference reduction uses a default padding of 3 detector pixels.

For a new field, use the correct region mask for that pointing and orientation.

## 16. Step 11 — SkyMapper photometric calibration

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py \
    --only 11 --run-qc
```

Step 11 calibrates against SkyMapper DR4 through VizieR catalog `II/379/smssdr4`, using a 1 arcsec nearest-neighbor match followed by robust clipping.

The SkyMapper query can be cached locally under:

```text
data/skymapper/<TARGET>/
```

The cache is optional. If it is absent, the pipeline can query through `astroquery`.

Inspect the QC for the number of matched stars, clipped outliers, zero points, residual scatter, and trends with magnitude or detector position.

## 17. Step 12 — PSF photometry

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py --only 12
```

Step 12 uses the established source list to obtain PSF-based photometry. The reference production configuration uses a Moffat model.

Products are written under:

```text
products/<TARGET>_<FIELD>/12_psfphot/
```

Compare PSF and aperture photometry before accepting the PSF measurements as the preferred flux estimator.

## 18. Step 13 — final calibrated catalog

Run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py \
    --only 13 --run-qc
```

The principal single-field product is:

```text
products/<TARGET>_<FIELD>/13_final_photometry/SISI_final_photometry.csv
```

Inspect calibrated magnitudes and uncertainties, color-magnitude behavior, PSF-versus-aperture residuals, spatial residuals, saturation/problem flags, and slit/bar flags.

## 19. Step 14 — merge multiple pointings

Step 14 is needed only when the same target has two independently reduced pointings such as T00 and T01.

Reduce both pointings through Step 13 first. Then run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_T00.py \
    --only 14 --run-qc
```

Outputs are written under:

```text
products/<TARGET>_merged_photometry/
```

including:

```text
SISI_merged_photometry.csv
SISI_overlap_matches_all.csv
SISI_overlap_bar_loss.csv
```

The merger preserves both measurements for overlap stars and uses DMD/slit and PSF-quality information to select the preferred measurement. It does not redo image reduction, source detection, or absolute photometric calibration.

## 20. After one successful field

Once a complete new-field reduction has been inspected and validated, the same configuration can be rerun in one command:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_MyTarget_T00.py \
    --from-step 00 --to-step 13 --run-qc
```

The driver is intended to make an already understood reduction reproducible. It is not a substitute for validating a new observing configuration.

## 21. Common first-run failures

**A configured FITS file cannot be found.** Check the project root, night name, frame ranges, filename pattern, and whether flats came from a different night.

**The image contains wrapped or implausible values.** Stop at Step 02. Check the signed-int16/unsigned-16 recovery and pedestal settings. Do not try to hide the problem with flat-field or photometric corrections.

**Short flats show a bright center.** Check shutter timing before interpreting the structure as detector sensitivity. A short iris-shutter exposure can produce a radial effective-exposure gradient.

**The detector halves disagree after flat-fielding.** Inspect the master flat and the configured post-flat right-half correction separately. Do not tune multiple corrections simultaneously without identifying the physical origin of the mismatch.

**Astrometry is globally shifted.** Treat a global shift/rotation/scale error as a linear-WCS problem. The SIP calibration describes distortion and should not be used as a free correction for a wrong field pointing.

**The raw FITS header contains a WCS but the Step 00 output WCS is wrong.** This is currently a known limitation: Step 00 flips the image without yet transforming the WCS. Use the validated Step 07c reference mechanism until the acquisition-WCS interface is implemented and verified.

**SkyMapper calibration cannot query the catalog.** Check network access and `astroquery`. The cached catalog is optional and may not exist on a first run.

**Slit/bar flags are wrong.** Verify that `SISI_PIXEL_REGION` belongs to the correct target, pointing, and post-Step-00 detector orientation.

## 22. What a normal user should edit

A new reduction should normally require changes only to:

```text
config/reductions/sisi_<Target>_<Field>.py
drivers/run_sisi_pipeline_<Target>_<Field>.py
```

plus target-specific region/calibration assets when needed.

Do not begin by modifying the algorithms under `pipeline/sisi/`. If a parameter differs from one observing run to another, first determine whether it belongs in the reduction configuration.
