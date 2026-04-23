# ART-Based Ultrasound Tomography Reconstruction

Implementation of the Algebraic Reconstruction Technique (ART) for 2D ultrasound 
tomography, developed as part of the MDM2 Group Project at the University of Bristol.

## Overview

This project builds a mathematical model of ultrasound wave attenuation through 
heterogeneous materials and reconstructs interior material maps from boundary 
measurements using the Kaczmarz iterative method. Performance is evaluated using 
dual metrics: Structural Similarity Index (SSIM) and Sobel Sharpness Preservation.

## Requirements

Install dependencies using lib_setup.py

## Repository Structure

- fixed_model.py        # Fan-beam setup, forward projection, matrix A construction, ART solver
- FBP_SSIM.py           # Iterative Filtered Backprojection with ramp filter and Hann window
- fbp_code.py           # Supporting FBP utilities
- Noise.py              # Gaussian noise sweep with SSIM and Sharpness evaluation
- noise_analysis.py     # Extended noise analysis scripts 
- Rays.py               # Ray density parameter sweep and optimal threshold identification
- Noiseless.py          # Master graph generation for noiseless parameter analysis
- ARTparam_ssim.py      # Beam subdivision sweep on the Shepp-Logan phantom
- fan_angle_analysis.py # Fan angle sensitivity analysis
- Shepp_import.py       # Shepp-Logan phantom import utility from skimage

## Reproducing Key Results

**Noiseless baseline reconstruction**

python models/Noiseless.py

**Noise sensitivity sweep**
python models/noise_analysis.py

**Ray density parameter sweep (Nr vs Nb)**
python models/Rays.py

**Shepp-Logan subdivision optimisation**
python models/ARTparam_ssim.py

**ART vs FBP comparison**
python models/FBP_SSIM.py

## Key Results

| Phantom | SSIM | Sharpness Preservation |
|---|---|---|
| Bone (noiseless) | >90% | 99.7% |
| Shepp-Logan | 61% | 75% |
| Industrial crack | 74.3% | 97.1% |

Optimal parameters: Nr = 180, Nb = 96, 20 iterations, 61 beam subdivisions.
Noise diagnostic threshold: 5% — beyond this SSIM degrades below the 60% benchmark.

## Authors

Eleni Adamou, Myles Anderson, Ella Kutova, Dom O'Neill, Anish Soobagrah
University of Bristol, MDM2 Group 04, April 2026
