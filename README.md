# TESS Exoplanet Transit Vetting

Deep Learning semester project focused on classification of
TESS transit candidates using astronomical light curves.

## Project structure

- scripts/audit - data validation and quality control
- scripts/download - data acquisition and diagnostics
- scripts/preprocessing - manifests and CNN input creation
- scripts/analysis - exploratory analysis and visualization
- scripts/training - dataset splitting and model training
- src/tess_vetting - reusable Python modules (planned)
- data - future normalized data layout
- reports - figures and experimental tables
- models - trained models and checkpoints
- docs - documentation and LaTeX report

## Current data location

Existing data are intentionally preserved:

- full_dataset/
- signals.csv
- lightcurves.csv
- targets.csv

Data migration is a separate planned step.

## Running scripts

Use the root-level launcher to preserve legacy paths.

List scripts:

    python run.py --list

Create CNN inputs:

    python run.py build_tess_inputs --per-class 0 --seed 42

Create TIC-based split:

    python run.py split_tess_dataset --seed 42

Train baseline:

    python run.py train_baseline --seed 42

## Labels

- 1 = confirmed planet
- 0 = false positive

Train/validation/test are grouped by TIC to prevent
the same star appearing across dataset splits.

## Status

Initial CNN input dataset and leakage-safe split are ready.
The preprocessing pipeline is still experimental and
requires further scientific validation.
