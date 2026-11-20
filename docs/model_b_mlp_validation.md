# Model B - Frozen Mantis-8M + MLP Head

## Method

- Pretrained encoder: `paris-noah/Mantis-8M` (frozen).
- Local and global embeddings concatenated into 1024 features.
- StandardScaler fitted only on training embeddings.
- MLP classifier: 1024 -> 128 -> 1.
- ReLU activation and dropout 0.3.
- Weighted BCEWithLogitsLoss.
- AdamW optimizer.
- Same TIC-based train/validation split as Model A.
- Checkpoint selected by validation Average Precision.
- Three random seeds: 42, 123, 2026.
- Test set not evaluated.

## Individual runs

| Seed | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3 | 0.9380 | 0.8750 | 0.7906 | 0.8382 | 1.14 |
| 123 | 4 | 0.9391 | 0.8803 | 0.7984 | 0.8435 | 1.15 |
| 2026 | 4 | 0.9414 | 0.8813 | 0.7909 | 0.8342 | 0.94 |

## Mean +/- sample standard deviation

| Metric | Mantis + MLP | Tuned CNN reference |
|---|---:|---:|
| average_precision | 0.9395 +/- 0.0018 | 0.9219 +/- 0.0127 |
| roc_auc | 0.8789 +/- 0.0034 | 0.8597 +/- 0.0054 |
| balanced_accuracy | 0.7933 +/- 0.0044 | 0.7685 +/- 0.0428 |
| f1 | 0.8386 +/- 0.0047 | 0.8131 +/- 0.0300 |
| accuracy | 0.7929 +/- 0.0052 | N/A |
| precision | 0.8908 +/- 0.0029 | N/A |
| recall | 0.7923 +/- 0.0077 | N/A |

## Limitations

- This is frozen-feature transfer learning, not end-to-end fine-tuning.
- The first MLP-head hyperparameters are provisional.
- Both models use validation-selected checkpoints.
- Three seeds do not represent independent data splits.
- Held-out test data remain unevaluated.
- Interpolation and observational artifacts may affect comparisons.

## Reproduction

```powershell
python -u .\scripts\training\train_mantis_head.py
```
