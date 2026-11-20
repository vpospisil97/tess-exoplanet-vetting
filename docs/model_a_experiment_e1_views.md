# Experiment E1 - Local vs Global Transit Views

## 1. Research question

Does the two-view TESS CNN benefit from information in both the local transit window and the global phase-folded light curve?

**Hypothesis:** using both views gives higher validation Average Precision than either view alone.

## 2. Controlled setup

- Same TIC-based train, validation, and test split.
- Same labels and preprocessing.
- Same CNN architecture: 79,937 trainable parameters.
- Learning rate: 0.001.
- Dropout: 0.3.
- Batch size: 32.
- Weight decay: 0.0001.
- Maximum epochs: 35.
- Early stopping patience: 7.
- Seeds: 42, 123, 2026.
- Checkpoint selection: validation AP.
- Classification threshold: 0.5.
- Test set was not evaluated.

### Input variants

- Both: original local and global views.
- Local only: global flux and coverage mask replaced by zeros.
- Global only: local flux and coverage mask replaced by zeros.

Disabled inputs are zeroed during both training and validation.

## 3. Individual results

| View | Seed | Epoch | AP | ROC-AUC | Balanced accuracy | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| both | 42 | 14 | 0.9282 | 0.8610 | 0.7921 | 0.8177 | 5.33 |
| both | 123 | 28 | 0.9073 | 0.8537 | 0.7191 | 0.8406 | 7.45 |
| both | 2026 | 11 | 0.9303 | 0.8644 | 0.7944 | 0.7811 | 3.84 |
| local_only | 42 | 14 | 0.8870 | 0.8045 | 0.5000 | 0.0000 | 4.42 |
| local_only | 123 | 19 | 0.8933 | 0.8054 | 0.5000 | 0.0000 | 5.39 |
| local_only | 2026 | 11 | 0.8944 | 0.8108 | 0.5000 | 0.0000 | 4.17 |
| global_only | 42 | 21 | 0.8519 | 0.7704 | 0.5000 | 0.0000 | 7.48 |
| global_only | 123 | 18 | 0.8699 | 0.7723 | 0.5644 | 0.2735 | 6.52 |
| global_only | 2026 | 18 | 0.8688 | 0.7893 | 0.5000 | 0.0000 | 6.51 |

## 4. Mean and sample standard deviation

| Metric | Both | Local only | Global only |
|---|---:|---:|---:|
| average_precision | 0.9219 +/- 0.0127 | 0.8916 +/- 0.0040 | 0.8635 +/- 0.0101 |
| roc_auc | 0.8597 +/- 0.0054 | 0.8069 +/- 0.0034 | 0.7773 +/- 0.0104 |
| balanced_accuracy | 0.7685 +/- 0.0428 | 0.5000 +/- 0.0000 | 0.5215 +/- 0.0372 |
| f1 | 0.8131 +/- 0.0300 | 0.0000 +/- 0.0000 | 0.0912 +/- 0.1579 |
| precision | 0.8890 +/- 0.0721 | 0.0000 +/- 0.0000 | 0.3048 +/- 0.5279 |
| recall | 0.7605 +/- 0.1065 | 0.0000 +/- 0.0000 | 0.0536 +/- 0.0928 |
| accuracy | 0.7656 +/- 0.0158 | 0.3208 +/- 0.0000 | 0.3538 +/- 0.0571 |

## 5. Paired AP differences

| Seed | Both - Local | Both - Global |
|---:|---:|---:|
| 42 | +0.0412 | +0.0763 |
| 123 | +0.0140 | +0.0374 |
| 2026 | +0.0359 | +0.0615 |

## 6. Limitations

- This is an input-information ablation.
- All CNN branches remain present, but one branch receives constant zero input in single-view variants.
- Zeroed inputs are an artificial experimental condition.
- This experiment does not establish physical causality.
- Model checkpoints are selected by validation AP.
- Hyperparameters were previously tuned on the same validation split.
- Reported results are not independent test estimates.
- Potential observational and processing artifacts require separate investigation.
- The held-out test set remains unused.

## 7. Reproduction

Run from the repository root:

```powershell
python -u .\scripts\training\experiment_e1_views.py
```
