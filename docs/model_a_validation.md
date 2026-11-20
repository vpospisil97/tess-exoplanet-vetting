# Model A - Two-view 1D CNN

## Validation results - three random seeds

This report was generated automatically from local experiment results.

**Task:** binary classification of TESS transit signals.

- Positive class (1): confirmed
- Negative class (0): false_positive
- Model selection metric: validation Average Precision
- Test set: not used for model selection or evaluation

## 1. Dataset

| Split | Signals | Unique TIC | Confirmed | False positive |
|---|---:|---:|---:|---:|
| train | 1335 | 1252 | 892 | 443 |
| validation | 293 | 268 | 199 | 94 |
| test | 290 | 269 | 195 | 95 |

Splits were verified to have no overlap at the TIC (star) level.

## 2. Model architecture

The model consists of two separate one-dimensional convolutional encoders.

- Local view: 101 time bins, flux and coverage mask
- Global view: 201 time bins, flux and coverage mask
- Convolutional layers: 32, 64, and 64 channels per encoder
- Pooling: adaptive average and maximum pooling
- Classification head: 256 -> 128 -> 1
- Trainable parameters: 79,937

The model is trained from random initialization.

Input transformation: `arcsinh(flux_ppt / 5)` with coverage masking.

## 3. Hyperparameters

- `epochs`: `35`
- `batch_size`: `64`
- `lr`: `0.001`
- `weight_decay`: `0.0001`
- `dropout`: `0.3`
- `patience`: `7`
- `device`: `cuda`

Optimizer: AdamW. Loss: weighted BCEWithLogitsLoss.

Positive-class weight is calculated from the training labels.

## 4. Individual runs

| Seed | Best epoch | Epochs run | AP | ROC-AUC | Balanced Accuracy | F1 | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 16 | 23 | 0.9000 | 0.8405 | 0.7684 | 0.7596 | 3.52 |
| 123 | 9 | 16 | 0.9107 | 0.8418 | 0.7561 | 0.7356 | 2.17 |
| 2026 | 14 | 21 | 0.9131 | 0.8230 | 0.7467 | 0.7085 | 2.79 |

Each checkpoint was selected by the highest validation Average Precision.

## 5. Mean and standard deviation

| Metric | Logistic regression baseline | CNN mean | CNN sample SD |
|---|---:|---:|---:|
| Average Precision | 0.7479 | 0.9079 | 0.0070 |
| ROC-AUC | 0.6415 | 0.8351 | 0.0105 |
| Balanced Accuracy | 0.6130 | 0.7571 | 0.0109 |
| F1-score | 0.6154 | 0.7346 | 0.0256 |
| Accuracy | 0.5734 | 0.7031 | 0.0205 |
| Precision | 0.7937 | 0.9333 | 0.0074 |
| Recall | 0.5025 | 0.6064 | 0.0377 |

The standard deviation describes variation across three CNN random seeds on the same validation split. The baseline was evaluated in one run.

## 6. Confusion matrices

| Seed | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|
| 42 | 84 | 10 | 71 | 128 |
| 123 | 85 | 9 | 78 | 121 |
| 2026 | 87 | 7 | 86 | 113 |

Confusion matrices use a fixed classification threshold of 0.5.

## 7. Reproduction

Run from the repository root, with the prepared dataset available locally.

```powershell
python -u .\run.py train_cnn --epochs 35 --batch-size 64 --lr 0.001 --patience 7 --seed 42 --device cuda
python -u .\run.py train_cnn --epochs 35 --batch-size 64 --lr 0.001 --patience 7 --seed 123 --device cuda
python -u .\run.py train_cnn --epochs 35 --batch-size 64 --lr 0.001 --patience 7 --seed 2026 --device cuda
```

Generate this report using:

```powershell
python .\scripts\analysis\generate_model_a_report.py
```

## 8. Limitations and next steps

- Results are validation results, not independent test-set estimates.
- The checkpoint is selected using the same validation set reported here.
- A separate hyperparameter search and final evaluation are still required.
- The source of predictive information must be checked for metadata, processing, or observational artifacts.
- The pretrained/adapted Model B has not yet been evaluated.
- The current results do not establish statistical significance versus the baseline.
