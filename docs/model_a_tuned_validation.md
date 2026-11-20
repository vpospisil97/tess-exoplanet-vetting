# Model A - Tuned CNN Validation

## 1. Hyperparameter selection

The configuration was selected from 27 grid-search trials using validation Average Precision.

- Selected trial: `lr_0p001__drop_0p3__batch_32`
- Learning rate: `0.001`
- Dropout: `0.3`
- Batch size: `32`
- Weight decay: `0.0001`
- Maximum epochs: `35`
- Patience: `7`
- Grid-search AP: `0.9282`

## 2. Repeated training

The selected configuration was retrained using three different random seeds.

| Seed | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 14 | 0.9282 | 0.8610 | 0.7921 | 0.8177 | 5.26 |
| 123 | 28 | 0.9073 | 0.8537 | 0.7191 | 0.8406 | 8.89 |
| 2026 | 11 | 0.9303 | 0.8644 | 0.7944 | 0.7811 | 4.35 |

## 3. Comparison with original CNN

Both configurations use the same TIC-based training and validation splits.

| Metric | Original mean +/- SD | Tuned mean +/- SD |
|---|---:|---:|
| average_precision | 0.9079 +/- 0.0070 | 0.9219 +/- 0.0127 |
| roc_auc | 0.8351 +/- 0.0105 | 0.8597 +/- 0.0054 |
| balanced_accuracy | 0.7571 +/- 0.0109 | 0.7685 +/- 0.0428 |
| f1 | 0.7346 +/- 0.0256 | 0.8131 +/- 0.0300 |
| accuracy | 0.7031 +/- 0.0205 | 0.7656 +/- 0.0158 |
| precision | 0.9333 +/- 0.0074 | 0.8890 +/- 0.0721 |
| recall | 0.6064 +/- 0.0377 | 0.7605 +/- 0.1065 |

## 4. Interpretation and limitations

- Average Precision is the primary model-selection metric.
- Both architectures are the same; only training hyperparameters differ.
- The held-out test split was not evaluated.
- The best configuration was chosen using the validation set.
- Seed 42 also participated in the original hyperparameter search.
- Repeated validation results measure initialization sensitivity, not independent generalization performance.
- Final performance must be evaluated on the held-out test data after model selection.
- Potential observational or processing artifacts still require investigation.

## 5. Reproduction

Run from the repository root:

```powershell
python .\scripts\training\grid_search_cnn.py
python .\scripts\training\repeat_best_cnn.py
```

Local experiment directory: `full_dataset/cnn_tuned_repeats/lr_0p001__drop_0p3__batch_32`
