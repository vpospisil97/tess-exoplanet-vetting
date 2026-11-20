# Model B - Validation Error Analysis

## Dataset and protocol

- 293 validation signals.
- 199 confirmed planets (positive class).
- 94 false positives (negative class).
- Same TIC-separated validation data for both models.
- Three random seeds per configuration.
- Decision threshold fixed at 0.5.
- Test set not used.

## Per-seed confusion matrices

| Model | Seed | AP | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 42 | 0.9380 | 0.8750 | 74 | 20 | 41 | 158 |
| Original | 123 | 0.9391 | 0.8803 | 75 | 19 | 40 | 159 |
| Original | 2026 | 0.9414 | 0.8813 | 75 | 19 | 43 | 156 |
| Tuned | 42 | 0.9414 | 0.8823 | 75 | 19 | 40 | 159 |
| Tuned | 123 | 0.9414 | 0.8839 | 71 | 23 | 39 | 160 |
| Tuned | 2026 | 0.9433 | 0.8833 | 76 | 18 | 43 | 156 |

## Persistent errors

Persistent error = a signal misclassified
in all three runs of the same configuration.

| Model | Persistent FP | Persistent FN | Total |
|---|---:|---:|---:|
| Original | 12 | 36 | 48 |
| Tuned | 15 | 35 | 50 |

## Changes after tuning

For each validation signal, the number of
incorrect predictions across three seeds
is compared between the original and tuned head.

- Improved (fewer incorrect runs): 11
- Worsened (more incorrect runs): 11
- Unchanged: 271

## Methodological cautions

- Errors are defined at a fixed 0.5 threshold.
- Average Precision does not depend on this threshold.
- Persistent errors may reflect ambiguous signals,
  difficult transit shapes, missing coverage,
  or incorrect labels; this audit cannot distinguish
  those explanations without inspecting the light curves.
- The validation set was used for model selection.
- No test examples were inspected.
