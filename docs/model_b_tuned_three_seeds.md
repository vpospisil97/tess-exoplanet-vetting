# Model B — Selected Mantis head, three random seeds

## Selected hyperparameters

- Selected by validation AP in the seed-42 nine-trial grid: `hidden_128__drop_0p5`
- Hidden width: 128
- Dropout: 0.5
- Pretrained Mantis-8M encoder frozen; only MLP head trained
- TIC-grouped train and validation data identical to original Model B
- Feature scaling fitted on training data only
- Seeds: 42, 123, 2026
- Held-out test data not loaded or evaluated

## Per-seed validation results

| Seed | Original AP | Tuned AP | Original ROC-AUC | Tuned ROC-AUC | Tuned F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9380 | 0.9414 | 0.8750 | 0.8823 | 0.8435 | 3 |
| 123 | 0.9391 | 0.9414 | 0.8803 | 0.8839 | 0.8377 | 4 |
| 2026 | 0.9414 | 0.9433 | 0.8813 | 0.8833 | 0.8365 | 4 |

## Three-seed validation mean ± sample standard deviation

| Metric | Original Mantis head | Grid-selected head | Difference in means |
|---|---:|---:|---:|
| average_precision | 0.9395 ± 0.0018 | 0.9420 ± 0.0011 | +0.0025 |
| roc_auc | 0.8789 ± 0.0034 | 0.8832 ± 0.0008 | +0.0043 |
| balanced_accuracy | 0.7933 ± 0.0044 | 0.7914 ± 0.0103 | -0.0019 |
| f1 | 0.8386 ± 0.0047 | 0.8392 ± 0.0038 | +0.0006 |

## Interpretation limits

- Seed 42 was used both for grid selection and in this repeat; its result is not independent of selection.
- All metrics come from the same validation stars used for model selection.
- Smaller standard deviation across initializations is not proof of generalization.
- Final comparison requires one-time evaluation on the held-out TIC-grouped test set.
- Feature extraction time is excluded from MLP training time.

## Reproduction

```powershell
python -u .\scripts\training\train_mantis_head_tuned.py
```
