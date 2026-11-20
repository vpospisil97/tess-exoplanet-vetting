# Experiment E2 - Coverage Mask Ablation

## 1. Research question

Does the explicit coverage mask improve the classification of TESS transit signals?

## 2. Hypothesis

Providing an explicit binary coverage channel improves validation Average Precision compared with using the flux channels alone.

## 3. Experimental setup

- Dataset: TESS transit signals.
- Same TIC-based data split.
- Same CNN architecture.
- Trainable parameters: 79,937.
- Same flux preprocessing.
- Learning rate: 0.001.
- Dropout: 0.3.
- Batch size: 32.
- Weight decay: 0.0001.
- Maximum epochs: 35.
- Early stopping patience: 7.
- Random seeds: 42, 123, 2026.
- Checkpoint selection: validation AP.
- Test set was not evaluated.

## 4. Experimental conditions

### With mask

The model receives both input channels:

- Transformed flux.
- Explicit binary coverage mask.

### Without mask

The flux channel remains unchanged.

The explicit coverage channel is replaced with zeros before convolutional encoding.

The same masking is applied during training and validation.

## 5. Individual results

| Mode | Seed | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| with_mask | 42 | 14 | 0.9282 | 0.8610 | 0.7921 | 0.8177 | 33.69 |
| with_mask | 123 | 28 | 0.9073 | 0.8537 | 0.7191 | 0.8406 | 14.82 |
| with_mask | 2026 | 11 | 0.9303 | 0.8644 | 0.7944 | 0.7811 | 7.06 |
| without_mask | 42 | 14 | 0.9255 | 0.8647 | 0.8116 | 0.8511 | 8.39 |
| without_mask | 123 | 7 | 0.9074 | 0.8310 | 0.7525 | 0.7529 | 6.01 |
| without_mask | 2026 | 17 | 0.9285 | 0.8639 | 0.7854 | 0.7943 | 8.41 |

## 6. Mean and sample standard deviation

| Metric | With mask | Without mask | Difference |
|---|---:|---:|---:|
| average_precision | 0.9219 +/- 0.0127 | 0.9205 +/- 0.0114 | +0.0014 |
| roc_auc | 0.8597 +/- 0.0054 | 0.8532 +/- 0.0193 | +0.0065 |
| balanced_accuracy | 0.7685 +/- 0.0428 | 0.7832 +/- 0.0296 | -0.0146 |
| f1 | 0.8131 +/- 0.0300 | 0.7994 +/- 0.0493 | +0.0137 |
| accuracy | 0.7656 +/- 0.0158 | 0.7588 +/- 0.0479 | +0.0068 |
| precision | 0.8890 +/- 0.0721 | 0.9108 +/- 0.0087 | -0.0218 |
| recall | 0.7605 +/- 0.1065 | 0.7152 +/- 0.0817 | +0.0452 |

## 7. Paired AP differences

| Seed | With mask | Without mask | Difference |
|---:|---:|---:|---:|
| 42 | 0.9282 | 0.9255 | +0.0027 |
| 123 | 0.9073 | 0.9074 | -0.0001 |
| 2026 | 0.9303 | 0.9285 | +0.0018 |

## 8. Interpretation and limitations

- Average Precision is the main metric.
- Both variants use identical data splits and training settings.
- Both variants have the same declared network architecture.
- The missing explicit coverage channel reduces usable input information.
- Missing bins may still be identifiable from zero-filled flux values.
- Therefore this is not complete removal of all missing-data information.
- Mask information may encode useful coverage or unwanted observational bias.
- The model was previously tuned on the same validation split.
- Three seeds measure sensitivity to initialization, not independent generalization performance.
- The held-out test data remains unused.

## 9. Reproduction

Run from repository root:

```powershell
python -u .\scripts\training\experiment_e2_coverage.py
```
