# SISI Astrometry: Operational WCS Contract

## Purpose

This note separates three concepts that must not be conflated:

1. the **field-specific preliminary/linear WCS**;
2. the **instrument distortion calibration** represented by SIP coefficients;
3. the geometric transformation applied by the reduction pipeline when the detector image is reoriented.

The distinction is essential for making future SISI reductions routine and reproducible.

## 1. Field-specific preliminary WCS

For normal SAMOS operations, the telescope target-acquisition system and SAMOS GUI should provide enough information to write a valid preliminary WCS into every SISI science FITS header.

That WCS is field-specific. It describes the observed pointing and should include at least the standard celestial WCS information needed to reconstruct the linear TAN mapping, such as:

```text
CTYPE1 / CTYPE2
CRVAL1 / CRVAL2
CRPIX1 / CRPIX2
CUNIT1 / CUNIT2
PC or CD matrix
RADESYS
```

The exact keyword implementation is an acquisition-software responsibility, but the resulting header must describe the **raw detector image actually stored in the FITS file**.

The acquisition WCS is expected to provide an approximate absolute pointing, scale, and orientation. It is not the same thing as the high-order SISI distortion solution.

## 2. Instrument SIP distortion

The calibrated SISI distortion is represented by the frozen SIP model:

```text
config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json
```

This is an instrument-level calibration derived and validated from Run 8 data. It is conceptually distinct from the pointing of any individual target.

The SIP calibration should normally be reused only while the optical/detector configuration remains compatible with the configuration in which it was derived. A new SIP fit should be triggered by evidence of changed distortion, not merely by a field pointing offset.

## 3. Current reference reduction

The Dolidze 25 reference reduction is frozen for reproducibility. It uses:

```text
config/calibrations/SISI_T00_ADOPTED_LINEAR_WCS.fits
config/calibrations/SISI_SIP_SOAR_RUN8_FINAL.json
```

The FITS file provides the adopted linear TAN WCS for the reference field; the JSON file supplies the frozen fifth-order SIP distortion.

Current Step 07c copies the adopted linear WCS from an immutable reference and then attaches the SIP calibration. It does not solve a new WCS.

This mechanism is correct for reproducing the Run 8 reference reduction.

## 4. Intended future observing workflow

The normal operational path for new SAMOS data should become:

```text
SOAR telescope pointing + rotator state
                 |
                 v
          SAMOS acquisition GUI
                 |
                 v
 raw SISI FITS with valid preliminary WCS
                 |
                 v
 Step 00 detector orientation
 AND corresponding WCS transformation
                 |
                 v
 field-specific linear WCS in oriented frame
                 |
                 +---- frozen SISI SIP calibration
                 |
                 v
          final TAN-SIP WCS
```

This removes the need to build a target-specific astrometric bootstrap product by hand for every routine observation.

## 5. Current Step 00 limitation

The present production Step 00 performs:

```python
data_out = np.fliplr(data)
```

and writes the copied FITS header to the output image.

It records:

```text
ORIENTED = True
FLIPX    = True
```

but it does **not currently transform an input celestial WCS** to account for the horizontal pixel reflection.

Therefore the following assumption is currently unsafe:

```text
raw FITS WCS -> Step 00 output WCS is automatically correct
```

It is not.

Until Step 00 is upgraded, the acquisition-header WCS should be regarded as an intended future operational input rather than the production astrometric solution used by the current Run 8 reference reduction.

## 6. Required acquisition-GUI behavior

The SAMOS acquisition system should eventually guarantee that every SISI acquisition/science FITS file:

- contains a valid preliminary celestial WCS;
- records the actual telescope pointing used for the exposure;
- uses the correct instrument/rotator geometry;
- contains enough provenance to identify the WCS as telescope/acquisition-derived;
- exposes a visible validation/status indicator to the observer;
- never silently writes an absent, stale, or obviously nonsensical WCS as if it were valid.

A simple operational status such as:

```text
WCS: VALID
WCS SOURCE: TELESCOPE/ACQUISITION
```

would make the observing state explicit.

## 7. Required Step 00 behavior

When the raw detector image is reflected or otherwise geometrically transformed, the WCS must be transformed consistently.

The contract is:

> **A WCS always describes the pixel array with which it is stored. If the pipeline transforms the pixel geometry, it must transform the WCS by the same mapping.**

For the current horizontal flip of an image of width `NAXIS1`, the pixel-coordinate transformation is equivalent to reversing the x coordinate. The production implementation should perform this through a WCS-aware operation or a carefully validated analytic header transformation, not by modifying individual keywords ad hoc.

The implementation should be validated by comparing sky coordinates of a grid of corresponding raw/oriented pixels before and after the transform.

## 8. Required preflight checks

Before future Step 07 processing relies on the acquisition WCS, the pipeline should verify that:

- the WCS keywords exist;
- the celestial axes are recognized by `astropy.wcs.WCS`;
- the field center is within a reasonable tolerance of the commanded target;
- pixel scale is compatible with SISI;
- the determinant/sign of the linear transform has the expected orientation;
- transformed corner coordinates are finite;
- the WCS is consistent after Step 00 orientation;
- the SIP reference is compatible with the science image shape.

A failed check should produce an explicit error or high-visibility warning rather than silently continuing with an invalid astrometric state.

## 9. Linear WCS errors versus SIP errors

This distinction should be preserved during diagnosis:

```text
global translation        -> linear WCS / pointing
global rotation           -> linear WCS / rotator geometry
global scale error        -> linear WCS / plate scale
smooth field distortion   -> SIP / optical distortion
```

Do not refit or distort the SIP model merely to remove a global acquisition pointing error.

## 10. Version/provenance requirement

The final calibrated FITS header should make it possible to determine:

- source of the preliminary WCS;
- whether Step 00 transformed it;
- SIP calibration file/version;
- whether the linear WCS was later refined;
- pipeline/reduction version that wrote the final WCS.

This is important because the SISI astrometric solution is the geometric connection between celestial coordinates, the imaging detector, and the DMD slit configuration.
