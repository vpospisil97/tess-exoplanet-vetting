# TESS Exoplanet Transit Vetting

Deep Learning semester project: binary classification of **known TESS transit signals** using phase-folded stellar light curves. The study compares a logistic-regression baseline, a custom convolutional neural network (Model A), and a frozen pretrained time-series encoder with a trained classification head (Model B).

**Status:** The original experiment, its one-time TIC-disjoint held-out test, and a supplementary five-fold, TIC-grouped cross-validation (E3) of the baseline, CNN, Mantis head, and E1/E2 ablations are complete. The supplementary CV was performed on development data after model selection; it does **not** replace the original test.

## Final report, presentation and CV results

All final deliverables are stored in [`docs/`](docs/):

| File | Purpose |
|---|---|
| [`docs/prezentace.pptx`](docs/prezentace.pptx) | Final 16-slide PowerPoint presentation (editable) |
| [`docs/TESS_zprava_LaTeX.pdf`](docs/TESS_zprava_LaTeX.pdf) | Final seven-page academic report (PDF) |
| [`docs/TESS_zprava_LaTeX.tex`](docs/TESS_zprava_LaTeX.tex) | Report source with the requested submission filename |
| [`docs/latex/main.tex`](docs/latex/main.tex) | Canonical LaTeX source in the existing project layout |
| [`docs/latex/main.pdf`](docs/latex/main.pdf) | Matching compiled report in the existing LaTeX directory |
| [`docs/e3_cross_validation.md`](docs/e3_cross_validation.md) | Full E3 numerical results, paired ablation comparisons and methodological limitations |

The `.tex` files contain the same source. For compilation from the repository layout, use `docs/latex/main.tex` with **XeLaTeX** (twice for references); when original figures under `docs/figures/` are unavailable, the source can render its fallback vector figures.

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

## Supplementary 5-fold group cross-validation (E3)

**Purpose:** E3 evaluates sensitivity to the choice of development stars. It reuses the original **1,628 train + validation signals** and leaves all **290 original held-out test signals** separate and unchanged. The five outer folds use `StratifiedGroupKFold` with groups defined by TIC, so the same star never appears in both the training and evaluation portion of a fold. The five evaluation folds contain 317–334 signals each.

The logistic-regression baseline is evaluated once per fold. CNN, the frozen Mantis-8M encoder plus trained head, and each ablation are evaluated for three random seeds (`42`, `123`, `2026`) per fold. Neural models select their epoch on an **inner TIC-separated validation partition**, then restart training on the entire outer training partition before evaluating the untouched outer validation fold. Standardization for baseline/Mantis is fitted within the appropriate outer training partition.

These are **retrospective robustness results**, not an independent, unbiased assessment of hyperparameter selection: hyperparameters were previously tuned on the original validation subset, which now belongs to the CV pool. The displayed sample SD is computed across the **five fold-wise means** (after averaging three seeds within each fold for neural models), not across 15 independent dataset splits.

### E3 model performance

| Model | Average Precision | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.8016 ± 0.0247 | 0.7114 ± 0.0394 | 0.6726 ± 0.0361 | 0.6756 ± 0.0612 |
| Custom CNN, both views, with mask | 0.9000 ± 0.0151 | 0.8321 ± 0.0240 | 0.7414 ± 0.0317 | 0.7784 ± 0.0507 |
| **Frozen Mantis-8M + MLP** | **0.9289 ± 0.0071** | **0.8797 ± 0.0109** | **0.7947 ± 0.0232** | **0.8452 ± 0.0158** |

Mantis achieves a higher mean AP than CNN in **each of the five outer folds**; this does not establish statistical significance on a new population of stars.

### E1 and E2 in E3

| Ablation | CV Average Precision |
|---|---:|
| E1 — both local and global views | 0.9000 ± 0.0151 |
| E1 — local only | 0.8855 ± 0.0266 |
| E1 — global only | 0.8729 ± 0.0085 |
| E2 — with coverage mask (same CNN reference runs) | 0.9000 ± 0.0151 |
| E2 — without explicit coverage mask | 0.9015 ± 0.0152 |

- **E1:** paired ΔAP for both minus local: `+0.0145 ± 0.0148`, with 12/15 positive seed-fold pairs; both minus global: `+0.0271 ± 0.0170`, with 13/15 positive pairs.
- **E2:** paired ΔAP for with-mask minus without-mask: `−0.0014 ± 0.0040`; with-mask wins in 8/15 pairs. Thus the small positive effect on the original validation split was **not** replicated consistently by CV.
- The E1 ablations zero the *unused CNN branch representation after the encoder*. The E2 ablation zeroes the mask channel. These are controlled architecture ablations, but the effective capacity of E1 branches differs and zero flux bins may indirectly reveal missing coverage.

Full results: [`docs/e3_cross_validation.md`](docs/e3_cross_validation.md). The original E1/E2 validations and all test scores above remain unchanged.

### E3 reproduction — existing processed data

These scripts require locally available `full_dataset/full_cnn_views.npz`, `full_dataset/cnn_split_indices.npz` and the previously extracted Mantis train/validation embeddings. These large data and checkpoint artifacts remain ignored by Git. **Do not generate the split again or run the original held-out test to reproduce E3.** From the repository root:

```powershell
# Prepare five TIC-grouped outer folds; original test is excluded.
python -u .\scripts\training\prepare_e3_folds.py

# Five baseline folds.
python -u .\scripts\training\cross_validate_baseline.py

# Five outer folds x three seeds each.
python -u .\scripts\training\cross_validate_cnn.py
python -u .\scripts\training\cross_validate_mantis.py

# E1 local only, E1 global only, E2 without mask; reuse CNN reference runs.
python -u .\scripts\training\cross_validate_ablations.py --experiment all

# Regenerate Markdown report from saved metrics.
python -u .\scripts\training\report_e3_full.py
```

The scripts support bounded pilots (`--max-folds 1 --max-seeds 1` for neural runs). They skip already completed fold/seed results and store outputs separately under `full_dataset/e3_cross_validation/`. They do **not** overwrite the original training, validation, E1/E2, or held-out-test artifacts.

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

The original training and hyperparameter-search scripts are in `scripts/training/`: `train_baseline.py`, `train_cnn.py`, `grid_search_cnn.py`, `repeat_best_cnn.py`, `extract_mantis_features.py`, `train_mantis_head.py`, `grid_search_mantis_head.py`, and `train_mantis_head_tuned.py`. The six supplementary E3 scripts are `prepare_e3_folds.py`, `cross_validate_baseline.py`, `cross_validate_cnn.py`, `cross_validate_mantis.py`, `cross_validate_ablations.py`, and `report_e3_full.py`.

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
  training/                  # baseline, CNN, Mantis, E1/E2, E3 grouped CV
  analysis/                  # visualization and error analysis
  evaluation/                # frozen final evaluation
docs/                        # final PPTX/PDF/TeX, CV report, experiment and test metrics
```

## Limitations and future work

- The labelled set is small and selected; results are not directly representative of real-world TESS candidate-vetting rates.
- The held-out test contains only 95 catalogue false positives. It was evaluated once on one TIC-disjoint split. E3 adds five development-only folds, not additional independent held-out tests.
- E3 was performed **after** model/hyperparameter selection on the original validation split, so it is a robustness check rather than an unbiased nested-CV estimate of the entire selection procedure.
- The positive-class prevalence in this curated test is high; AP should be interpreted relative to that prevalence.
- Classification scores are **not calibrated probabilities** of planet confirmation.
- TIC grouping limits star overlap within this dataset but cannot exclude every form of source, survey or pretrained-encoder bias.
- The original HF download URLs use `resolve/main` rather than a pinned source revision, and the current requirements are not a complete dependency lock. Exact source snapshots and data provenance should be archived for stronger reproduction guarantees.
- The pilot manifest is optional for reconstruction. This README documents the original pipeline but does not claim an end-to-end clean-room rerun has already been performed.
- Future independent work may rank unconfirmed candidates for follow-up, provided the outputs are not presented as confirmed-planet probabilities or ground-truth evaluation.

## AI assistance

Generative AI supported code drafting, debugging, experiment planning and documentation. Generated outputs were reviewed against source files, executed tests, audited dataset artifacts and recorded metrics. Scientific conclusions and the final report require independent verification before submission.
