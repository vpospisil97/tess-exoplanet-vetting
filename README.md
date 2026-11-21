
# TESS Exoplanet Transit Vetting

Deep Learning semester project: binary vetting of TESS transit signals from phase-folded stellar light curves. The goal is to distinguish **confirmed planets (label 1)** from **catalogue false positives (label 0)**, and to compare a simple baseline, a custom convolutional network, and a pretrained time-series encoder.

**Status:** Main experimental study completed, including a one-time evaluation on a held-out TIC-disjoint test set. Candidate-only signals are *not* treated as confirmed planets in this study.

## Data and task

Source dataset: [saadtaleb/tess-lightcurves-planets](https://huggingface.co/datasets/saadtaleb/tess-lightcurves-planets). Source metadata include `signals.csv`, `targets.csv`, and `lightcurves.csv`, along with TESS FITS light curves. Inspect the original source and its licensing/provenance before redistributing data.

- Original metadata: 13,254 catalogue signals (9,895 candidates, 1,312 confirmed planets, 666 false positives, and 1,381 unknown).
- This **binary labelled** experiment uses 1,918 usable confirmed/false-positive signals after preprocessing. The 9,895 unconfirmed candidates are excluded from binary supervised training and evaluation.
- Input views: **local** `(101, 2)` and **global** `(201, 2)`, with relative flux and a coverage-mask channel.
- Positive class (`1`): confirmed planet; negative class (`0`): false positive.
- Data are split by **TIC star identifier**, not by FITS file or individual transit, to avoid sharing stars between train, validation, and test.

| Split | Signals | Unique TIC stars |
|---|---:|---:|
| Training | 1,335 | 1,252 |
| Validation | 293 | 268 |
| Held-out test | 290 | 269 |
| **Total** | **1,918** | **1,789** |

All 1,918 samples belong to exactly one split. An integrity audit found no overlap of TIC identifiers between splits.

> **Data availability:** Large data, FITS files, generated `.npz` files, and trained checkpoints live in the local `full_dataset/` directory and are intentionally excluded from Git. A fresh clone does **not** include the experiment dataset or weights. The acquisition and preprocessing scripts and original upstream dataset are required to rebuild them. The published test scores are for the frozen dataset/split described in `docs/pretest_audit.md`.

## Models and preprocessing

| Model | Description | Input / preprocessing |
|---|---|---|
| Baseline | Standard-scaled logistic regression | Flatten both raw preprocessed views (including masks), 604 features |
| Model A — CNN | Custom two-branch 1D CNN, trained from random initialization; 79,937 parameters | Flux transformed using `arcsinh(flux * 1000 / 5)` plus coverage mask |
| Model B — Mantis-8M | Frozen pretrained `paris-noah/Mantis-8M` encoder (8,103,936 parameters) plus trained MLP head (131,329 parameters) | Flux only; `arcsinh(flux / 5)`, interpolate local/global views separately to 512 samples, concatenate 512-dimensional embeddings |

For Model B, the embedding scaler is fitted **on training data only** and stored with the MLP head checkpoint. The pretrained encoder is **not fine-tuned**. Do not use one model's preprocessing for another model.

## Final held-out test results

The hyperparameters and checkpoints were selected with validation data before the held-out test was evaluated **once**. Primary metric: **Average Precision (AP)**. Values are mean ± sample standard deviation across seeds `42`, `123`, `2026` for the neural networks; the baseline has one run.

| Model | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7981 | 0.6979 | 0.6541 | 0.6189 |
| Custom CNN | 0.8851 ± 0.0049 | 0.8225 ± 0.0014 | 0.7308 ± 0.0107 | 0.7772 ± 0.0679 |
| Frozen Mantis-8M + MLP | **0.9009 ± 0.0024** | **0.8434 ± 0.0018** | **0.7539 ± 0.0144** | **0.8119 ± 0.0106** |

The test set contains **195 confirmed** and **95 false-positive** signals; this is a curated binary subset rather than a representative sample of all TESS detections. These results do not establish the probability that an unconfirmed candidate is a planet. Standard deviations are across model random seeds on **one fixed test split**, not across independent dataset resamplings.

Detailed results: [`docs/final_test_results.md`](docs/final_test_results.md).

## Environment

Development and evaluation used:

- Windows 11, Python **3.10.6**
- NVIDIA GeForce RTX **4060 Laptop GPU**, CUDA enabled
- PyTorch **2.14.1+cu130**, NumPy **2.2.6**, pandas **2.3.3**, scikit-learn **1.7.2**, Astropy **6.1.7**
- Mantis package: `mantis-tsfm` (the project used version **1.1.0**)
- Seeds: **42, 123, 2026**; original TIC split created with seed **42**

Create and activate a virtual environment (PowerShell):

```powershell
cd C:\path\to\tess-exoplanet-vetting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To reproduce **CUDA** measurements, install a compatible CUDA-enabled PyTorch build for your GPU and driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). The generic `requirements.txt` cannot guarantee the exact CUDA build on every system. Mantis encoder extraction in the experiment requires CUDA; the final evaluator also supports selecting CPU if dependencies permit. Package versions above document the original run; the ranges in `requirements.txt` are intended for installation rather than an exact environment lock.

## Project structure

```text
README.md
requirements.txt
run.py                         # root launcher for older scripts
signals.csv                    # local metadata (not committed)
targets.csv                    # local metadata (not committed)
lightcurves.csv                # local metadata (not committed)
full_dataset/                  # local processed data, checkpoints, predictions (ignored)
scripts/
  audit/                       # data checks and leakage audits
  download/                    # TESS acquisition and download diagnostics
  preprocessing/               # manifest and phase-folded input preparation
  training/                    # baseline, CNN, Mantis and experiments
  analysis/                    # error analysis and visualizations
  evaluation/                  # frozen final evaluation

docs/                         # experiment reports, figures, final metrics
```

Some older scripts expect paths supplied by `run.py`. Run them via the root launcher; newer experiment scripts explicitly resolve the repository root.

## Reproducing the experiment

**Do not rerun the already completed held-out test to search for better scores.** The commands below describe the original experimental pipeline for a **fresh reproduction environment** with the original data available.

1. Obtain the upstream dataset and its light curves. Place source metadata at the project root and the required local FITS/data under the paths expected by the download/preprocessing scripts. Check `python run.py --list` for supported root-launcher tasks. Acquisition needs substantial disk space; the full dataset is not bundled with the repository.
2. Prepare the 1,918-signal binary dataset and create a TIC-grouped split:

   ```powershell
   python run.py build_tess_inputs --per-class 0 --seed 42
   python run.py split_tess_dataset --seed 42
   ```

   Expected files: `full_dataset/full_cnn_views.npz` and `full_dataset/cnn_split_indices.npz`. Consult the preprocessing scripts for any prerequisite manifests/other staging files. The original split fingerprints are recorded in `docs/pretest_audit.md`.
3. Train the baseline (train/validation only), then run the CNN hyperparameter grid and repeat its selected configuration:

   ```powershell
   python run.py train_baseline --seed 42
   python -u .\scripts\training\grid_search_cnn.py
   python -u .\scripts\training\repeat_best_cnn.py
   ```

4. Extract the pretrained Mantis embeddings for train/validation, train its original head, run the nine-trial head grid, and repeat the chosen head for three seeds:

   ```powershell
   python -u .\scripts\training\extract_mantis_features.py
   python -u .\scripts\training\train_mantis_head.py
   python -u .\scripts\training\grid_search_mantis_head.py
   python -u .\scripts\training\train_mantis_head_tuned.py
   ```

   The pretrained checkpoint `paris-noah/Mantis-8M` may need to be downloaded from Hugging Face on first use.
5. Verify split integrity and inspect validation-only results:

   ```powershell
   python -u .\scripts\audit\audit_pretest.py
   ```

   The final evaluator has two distinct modes:

   ```powershell
   python -u .\scripts\evaluation\evaluate_tess_final.py --preflight
   # --test is reserved for a first, pre-registered held-out test in a new reproduction.
   ```

   `--preflight` checks saved validation predictions and Mantis embeddings. **In the completed original run**, `full_dataset/final_test/test_started.json` already exists and prevents a second preflight/test run; do not remove this lock to retune models against the same test data. The `--test` mode was run exactly once, after validation checks passed.

The frozen evaluator additionally verifies SHA-256 fingerprints of the input arrays and split file, so newly rebuilt datasets may not have matching hashes even when procedurally equivalent. A fresh exact reproduction requires the same input artifacts, software setup, and selected checkpoints; document any new dataset/split as a separate experiment rather than overwriting the published test.

## Experiments and reports

- CNN tuning: [`docs/model_a_grid_search.md`](docs/model_a_grid_search.md), [`docs/model_a_tuned_validation.md`](docs/model_a_tuned_validation.md)
- CNN ablations: `docs/model_a_experiment_e1_features.md`, `docs/model_a_experiment_e2_coverage.md`
- Mantis feature extraction and tuning: `docs/model_b_feature_extraction.md`, `docs/model_b_head_grid_search.md`, `docs/model_b_tuned_three_seeds.md`
- Error analysis: `docs/model_b_error_analysis.md`, `docs/model_b_error_examples.md`, `docs/model_b_metadata_audit.md`
- Pretest audit: [`docs/pretest_audit.md`](docs/pretest_audit.md)
- Final held-out test: [`docs/final_test_results.md`](docs/final_test_results.md)

## Limitations and future work

- Only 1,918 eligible **binary-labelled** signals; the catalogue's far larger *candidate* class has no trusted planet/non-planet ground truth for this experiment.
- Selection effects, class balance and catalogue labels limit generalization to new candidates and detection pipelines.
- The TIC-grouped split prevents within-dataset star overlap, but cannot rule out all data-source, survey-selection or external-pretraining biases.
- Threshold-based metrics use a fixed decision threshold of **0.5**. Model scores should not be interpreted as calibrated planet probabilities.
- The frozen Mantis model has been pretrained on other time-series data; results are not a comparison between models trained from identical initial knowledge.
- Future work: independent catalogue validation, larger high-quality labelled datasets, uncertainty/calibration assessment, and **separate inference-only exploration of unconfirmed candidates** without assigning them ground-truth labels.

## AI assistance

Generative AI was used to assist with code drafting, debugging, documentation, and experiment planning. Outputs were reviewed against the actual source files and validated by execution, dataset audits, checkpoints, and recorded metrics. AI-generated text and scientific interpretation require independent checking before submission.
