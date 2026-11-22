# TESS Exoplanet Transit Vetting

Deep Learning semester project: binary classification of **known TESS transit signals** using phase-folded stellar light curves. The study compares a logistic-regression baseline, a custom convolutional neural network (Model A), and a frozen pretrained time-series encoder with a trained classification head (Model B).

**Status:** The original experiment is complete. Final model checkpoints were chosen on validation data; the TIC-disjoint held-out test set was evaluated once, without subsequent retuning.

## Research question and labels

Can a classifier distinguish **catalogue-confirmed planets** (label `1`) from **catalogue false positives** (label `0`) using local and global views of a light curve?

This is **not** the same as confirming previously unknown planets. Unconfirmed `candidate` and `unknown` entries do not provide reliable binary ground-truth labels and are excluded from this supervised experiment.

## Dataset

Original source: [saadtaleb/tess-lightcurves-planets](https://huggingface.co/datasets/saadtaleb/tess-lightcurves-planets).

| Catalogue label | Original count | In the final binary dataset |
|---|---:|---:|
| `confirmed` | 1,312 | 1,286 |
| `false_positive` | 666 | 632 |
| `candidate` | 9,895 | Not used |
| `unknown` | 1,381 | Not used |
| **Total** | **13,254** | **1,918** |

### Selection lineage

The original catalogue has 1,978 binary-labelled signals. The dataset was built as follows, using the saved selection reports:

| Selection stage | Removed at stage | Remaining |
|---|---:|---:|
| Catalogue `confirmed` + `false_positive` | — | 1,978 |
| `preliminary_ready`: source readiness, valid ephemeris and at least one usable audited FITS file | 39 | 1,939 |
| Transit coverage check | 7 | 1,932 |
| Transit duration / orbital period must be `< 0.25` | 1 | 1,931 |
| Sufficient local and global phase-bin coverage | 13 | **1,918** |

The 39 preliminary exclusions may have multiple contributing causes; they have **not yet been individually attributed** to the readiness, ephemeris and FITS conditions. Coverage requires at least five time samples within transit and ten nearby out-of-transit samples. The final phase-bin filter requires at least 45% covered local bins, 30% covered global bins and five covered central local bins.

See [`docs/dataset_selection_lineage.md`](docs/dataset_selection_lineage.md) and the generating script `scripts/audit/audit_selection_lineage.py`.

### Leakage-safe split

Each TIC star belongs to exactly one subset, including stars containing more than one signal. The split groups stars by `negative_only`, `positive_only`, and `mixed` before distributing groups with seed `42`.

| Split | Signals | Unique TIC stars |
|---|---:|---:|
| Train | 1,335 | 1,252 |
| Validation | 293 | 268 |
| Held-out test | 290 | 269 |
| **Total** | **1,918** | **1,789** |

The saved audit verified no TIC overlap between splits. Local model input: `(101, 2)`; global model input: `(201, 2)`. The two channels are relative flux and observed-bin coverage (mask).

## Models

| Model | Architecture | Inputs and preprocessing |
|---|---|---|
| Baseline | Standardized logistic regression | Flattened local and global views; 604 features |
| Model A | Custom two-branch 1D CNN, **79,937 parameters**, trained from scratch | `arcsinh(flux * 1000 / 5)`, plus coverage masks |
| Model B | Frozen [`paris-noah/Mantis-8M`](https://huggingface.co/paris-noah/Mantis-8M) encoder (**8,103,936 parameters**) + trained MLP head (**131,329 parameters**) | `arcsinh(flux / 5)`; local and global views interpolated separately to 512 positions; embeddings concatenated |

The Model B embedding scaler is fitted **only on training embeddings** and saved with the classification head. The pretrained encoder itself was not fine-tuned. Model-specific preprocessing must not be interchanged.

## Final held-out test

The primary metric is **Average Precision (AP)**. Model selection used validation AP. The test comprises **195 confirmed** and **95 false-positive** signals. A `0.5` classification threshold was used for F1 and balanced accuracy. Neural-model entries are mean ± sample standard deviation for seeds `42`, `123`, `2026` on the **same fixed split**.

| Model | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7981 | 0.6979 | 0.6541 | 0.6189 |
| Custom CNN | 0.8851 ± 0.0049 | 0.8225 ± 0.0014 | 0.7308 ± 0.0107 | 0.7772 ± 0.0679 |
| Frozen Mantis-8M + MLP | **0.9009 ± 0.0024** | **0.8434 ± 0.0018** | **0.7539 ± 0.0144** | **0.8119 ± 0.0106** |

See [`docs/final_test_results.md`](docs/final_test_results.md) for per-seed results, confusion matrices and runtime. The mean differences should not be interpreted as formal proof of statistical superiority across independent datasets.

## Environment

Original experiment used Windows 11, Python **3.10.6**, NVIDIA GeForce **RTX 4060 Laptop GPU** and CUDA. Recorded library versions included PyTorch **2.14.1+cu130**, NumPy **2.2.6**, pandas **2.3.3**, scikit-learn **1.7.2**, Astropy **6.1.7**, and `mantis-tsfm==1.1.0`.

On Windows PowerShell:

```powershell
cd C:\path\to\tess-exoplanet-vetting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file uses some version ranges and is **not** an exact environment lock. For GPU reproduction, install a driver-compatible CUDA-enabled PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/); the generic requirements file does not guarantee the original CUDA variant.

## Data acquisition and preprocessing — original pipeline

**Important:** The commands below describe the pipeline for a **fresh, separate reconstruction**. They are **not instructions to rerun preprocessing in the existing finished experiment**, where doing so would overwrite frozen `.npz`, split and intermediate CSV artifacts.

### Inputs needed before running

- Upstream `signals.csv`, `lightcurves.csv` and `targets.csv` at repository root; `signals.csv` must include the source's `training_ready_minimum` and transit ephemerides.
- Optional pilot manifest: `TESS_sample_150/tess_sample_150_manifest.csv` with `tic_id` and `hf_path`. If present, `build_full_manifest.py` marks matching FITS as available in the pilot; if absent, it builds the full download manifest with `in_pilot = False`. The pilot manifest is not required for a fresh clone.
- Network access and ample disk space for the FITS dataset. `download_full_dataset.py` defaults to keeping **40 GiB free disk reserve**.

For legacy scripts, use `python run.py SCRIPT_NAME` from the repository root. The launcher emulates the original root-level file location, which is needed because those scripts use paths relative to `__file__`.

### Original execution order

```powershell
# 1. Select valid source metadata and prepare FITS download URLs.
python run.py build_full_manifest

# 2. Download FITS light curves (potentially thousands of files / large disk use).
python run.py download_full_dataset --limit 0

# 3. Audit downloaded FITS files.
python run.py audit_fits_quality --mode full

# 4. Join catalogue signals with audited FITS paths.
python run.py build_training_manifest

# 5. Check transit observation coverage.
python run.py audit_transit_coverage

# 6. Optional diagnostic report; does not control the main filter.
python run.py review_transit_coverage

# 7. Build two-view CNN input arrays from all eligible labelled signals.
python run.py build_tess_inputs --per-class 0 --seed 42

# 8. Split complete TIC star groups into train / validation / test.
python run.py split_tess_dataset --seed 42
```

Expected intermediate outputs, all under `full_dataset/`:

| Script | Main output |
|---|---|
| `build_full_manifest` | `fits_download_manifest.csv`, `eligible_signals.csv` |
| `download_full_dataset` | `fits/` (downloaded FITS), `download_log.csv` |
| `audit_fits_quality --mode full` | `fits_quality_full.csv` |
| `build_training_manifest` | `training_manifest.csv`, `training_fits_links.csv` |
| `audit_transit_coverage` | `transit_coverage_detail.csv`, `transit_coverage_summary.csv` |
| `build_tess_inputs` | `full_cnn_views.npz`, `full_cnn_qc.csv` |
| `split_tess_dataset` | `cnn_split_indices.npz`, `cnn_split_manifest.csv`, `cnn_split_summary.csv` |

The two required phase-folded views are generated after filtering nonfinite and nonpositive flux, requiring `QUALITY == 0` where available, normalizing per FITS light curve, removing duplicate timestamps, and median-binning the folded phase. Observed-bin coverage is retained as a separate channel.

The legacy `prepare_lightcurves.py` is a **separate pilot preprocessing path**, not part of the full 1,918-signal pipeline. `analyze_fits_quality.py` and `check_quality_flags.py` are diagnostics, not substitutes for `audit_fits_quality --mode full`.

### Read-only audits of the finished data

From the existing original workspace (with all saved `full_dataset/` artifacts):

```powershell
# These audit the existing files without rebuilding the model inputs.
python -u .\scripts\audit\audit_selection_lineage.py
python -u .\scripts\audit\audit_pretest.py
```

`audit_selection_lineage.py` uses a direct path to the repository root; **do not invoke it through the legacy `run.py` launcher**. The pretest audit checks model artifacts and the TIC-based split. It does not evaluate held-out test predictions.

## Training and evaluation

The trained model and hyperparameter-search scripts are in `scripts/training/`: `train_baseline.py`, `train_cnn.py`, `grid_search_cnn.py`, `repeat_best_cnn.py`, `extract_mantis_features.py`, `train_mantis_head.py`, `grid_search_mantis_head.py`, and `train_mantis_head_tuned.py`.

The original evaluation script is `scripts/evaluation/evaluate_tess_final.py`. Its preflight mode verifies validation predictions and frozen checkpoints. The held-out `--test` mode was already run **exactly once** in the original experiment; it must not be used for further model or threshold selection. See `docs/pretest_audit.md` and `docs/final_test_results.md` instead of rerunning the published test.

Reconstructing the *exact* final scores also requires the original processed-data bytes, model checkpoints, preprocessing setup and split indices; all of these are local artifacts not stored in Git. The final evaluator validates hashes of the original dataset and split. Recreated data with different hashes must be treated as a separate run, not silently substituted into the original results.

## Repository layout

```text
README.md
requirements.txt
run.py
signals.csv                  # local source data; ignored by Git
lightcurves.csv              # local source data; ignored by Git
targets.csv                  # local source data; ignored by Git
TESS_sample_150/             # local pilot dataset and manifest; ignored by Git
full_dataset/                # local FITS, processed arrays, checkpoints; ignored
scripts/
  audit/                     # FITS audits, transit coverage, integrity
  download/                  # acquisition and download diagnostics
  preprocessing/             # manifests and phase-folded views
  training/                  # baseline, CNN, Mantis and experiments
  analysis/                  # visualization and error analysis
  evaluation/                # frozen final evaluation
docs/                        # experiment reports and final test metrics
```

## Limitations and future work

- The labelled set is small and selected; results are not directly representative of real-world TESS candidate-vetting rates.
- The test contains only 95 catalogue false positives. Evaluation uses one TIC-disjoint split, not independent split replications.
- The positive-class prevalence in this curated test is high; AP should be interpreted relative to that prevalence.
- Classification scores are **not calibrated probabilities** of planet confirmation.
- TIC grouping limits star overlap within this dataset but cannot exclude every form of source, survey or pretrained-encoder bias.
- The original HF download URLs use `resolve/main` rather than a pinned source revision, and the current requirements are not a complete dependency lock. Exact source snapshots and data provenance should be archived for stronger reproduction guarantees.
- The pilot manifest is optional. However, an end-to-end clean-clone reproduction has not yet been performed. Upstream metadata, FITS files, processed arrays and model checkpoints must be obtained or rebuilt separately.
- Future independent work may rank unconfirmed candidates for follow-up, provided the outputs are not presented as confirmed-planet probabilities or ground-truth evaluation.

## AI assistance

Generative AI supported code drafting, debugging, experiment planning and documentation. Generated outputs were reviewed against source files, executed tests, audited dataset artifacts and recorded metrics. Scientific conclusions and the final report require independent verification before submission.
