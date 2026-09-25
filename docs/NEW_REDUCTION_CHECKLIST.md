# New SISI Reduction Checklist

Use this checklist when reducing a new SISI field for the first time.

## Before running the pipeline

- [ ] Raw SISI FITS files are copied to the observing-data area and remain read-only.
- [ ] `SAMOS_PROJECT_ROOT` points to the project containing the observing run.
- [ ] Science night and flat night have been identified.
- [ ] Bias frame numbers have been checked.
- [ ] Science frame numbers have been checked for each filter.
- [ ] Exposure times have been checked.
- [ ] Flat frame numbers and exposure times have been checked.
- [ ] Target center and field identifier are correct.
- [ ] A new reduction configuration has been copied from a validated reference file.
- [ ] A new driver points to that reduction configuration.
- [ ] Detector-format/pedestal settings have been reviewed for this observing run.
- [ ] Post-flat right-amplifier corrections have not simply been copied without validation.

## Reduction

```text
00  Orient raw frames
01  Build superbias
02  Detector preprocessing
03  Median coaddition
04  Build master flats
05  Apply flats
07c WCS + SIP
09  Detection / aperture photometry
11  SkyMapper calibration
12  PSF photometry
13  Final catalog
14  Optional multi-pointing merge
```

### Step 00

- [ ] All expected configured frames were found.
- [ ] No zero-byte/corrupt FITS files.
- [ ] Images have the expected dimensions.
- [ ] Orientation is correct.
- [ ] WCS caveat understood: current Step 00 flips data but does not yet transform an input WCS.

### Step 01

- [ ] Superbias level is plausible.
- [ ] Both amplifier halves are understood.
- [ ] No unexpected strong structures.

### Step 02

- [ ] Science integer-format recovery is correct.
- [ ] Any pedestal-aware recovery is justified for this run.
- [ ] Flat-field processing has not inherited science pedestal subtraction.
- [ ] Flat-field processing has not erased the detector response.
- [ ] Pixel ranges and saturation behavior are plausible.

### Step 03

- [ ] Correct science frames entered each coadd.
- [ ] Stellar images remain sharp.
- [ ] No unexplained clipping or dynamic-range loss.

### Step 04

- [ ] Correct flats were used for each science band.
- [ ] Master flats are finite over the useful detector.
- [ ] Large-scale illumination/shutter effects are understood.
- [ ] Amplifier response has not been artificially forced away.

### Step 05

- [ ] Flat-fielded sky is sensible.
- [ ] Residual left/right amplifier mismatch is quantified.
- [ ] Any configured post-flat offset/scale is justified.

### Step 07c

- [ ] The adopted linear WCS reference is appropriate for this field.
- [ ] The SISI SIP calibration corresponds to the active instrument configuration.
- [ ] WCS overlays external stars correctly.
- [ ] Global pointing errors are treated as linear-WCS errors, not SIP errors.

### Step 09

- [ ] Detection band is appropriate.
- [ ] Source centroids look correct.
- [ ] Multi-band projected positions are correct.
- [ ] Slit/bar region file belongs to the correct pointing and orientation.
- [ ] Step 09 QC inspected.

### Step 11

- [ ] SkyMapper matches are plausible.
- [ ] Enough calibration stars survived clipping.
- [ ] Zero-point scatter is acceptable.
- [ ] No obvious residual trend with magnitude or position.
- [ ] Step 11 QC inspected.

### Step 12

- [ ] PSF model represents the stellar profiles.
- [ ] PSF and aperture photometry have been compared.
- [ ] Strong outliers are understood.

### Step 13

- [ ] Final calibrated catalog exists.
- [ ] Magnitude uncertainties are plausible.
- [ ] Color-magnitude behavior is sensible.
- [ ] Slit/bar and PSF-quality flags propagated correctly.
- [ ] Step 13 QC inspected.

### Step 14, if applicable

- [ ] Each pointing independently passed Step 13.
- [ ] Overlap matches are correct.
- [ ] Relative correction uses clean overlap stars.
- [ ] Barred/clean pairs behave as expected.
- [ ] Step 14 QC inspected.

## Only after validation

Once one complete field has passed these checks, it is reasonable to use the driver for a routine full run:

```bash
PYTHONPATH=. python drivers/run_sisi_pipeline_<Target>_<Field>.py \
    --from-step 00 --to-step 13 --run-qc
```
