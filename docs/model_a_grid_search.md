# Model A - CNN hyperparameter grid search

## Search design

- 27 combinations: learning rate x dropout x batch size.
- Learning rates: 0.0003, 0.001, 0.003.
- Dropout values: 0.1, 0.3, 0.5.
- Batch sizes: 32, 64, 128.
- Fixed seed: 42; epochs: 35; patience: 7; weight decay: 0.0001.
- Checkpoint selection and ranking: validation Average Precision (AP).
- The original three CNN seed runs are not overwritten.
- The independent test split was not evaluated.

## Ranked results

| Rank | Learning rate | Dropout | Batch | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.3 | 32 | 14 | 0.9282 | 0.8610 | 0.7921 | 0.8177 |
| 2 | 0.003 | 0.5 | 32 | 16 | 0.9232 | 0.8446 | 0.7607 | 0.7956 |
| 3 | 0.003 | 0.3 | 32 | 16 | 0.9217 | 0.8529 | 0.7497 | 0.7544 |
| 4 | 0.003 | 0.1 | 32 | 14 | 0.9216 | 0.8540 | 0.7627 | 0.8076 |
| 5 | 0.003 | 0.3 | 64 | 24 | 0.9204 | 0.8641 | 0.7628 | 0.8426 |
| 6 | 0.003 | 0.1 | 128 | 14 | 0.9194 | 0.8483 | 0.6892 | 0.6007 |
| 7 | 0.003 | 0.1 | 64 | 10 | 0.9146 | 0.8379 | 0.7665 | 0.7455 |
| 8 | 0.003 | 0.3 | 128 | 19 | 0.9136 | 0.8492 | 0.7606 | 0.7537 |
| 9 | 0.001 | 0.5 | 32 | 21 | 0.9099 | 0.8613 | 0.7814 | 0.8132 |
| 10 | 0.003 | 0.5 | 64 | 14 | 0.9089 | 0.8346 | 0.7497 | 0.7544 |
| 11 | 0.001 | 0.1 | 32 | 12 | 0.9070 | 0.8578 | 0.7987 | 0.8396 |
| 12 | 0.001 | 0.5 | 128 | 19 | 0.9068 | 0.8302 | 0.7602 | 0.8043 |
| 13 | 0.001 | 0.5 | 64 | 18 | 0.9043 | 0.8572 | 0.7795 | 0.8011 |
| 14 | 0.001 | 0.3 | 64 | 16 | 0.9000 | 0.8405 | 0.7684 | 0.7596 |
| 15 | 0.001 | 0.1 | 64 | 15 | 0.8982 | 0.8433 | 0.7137 | 0.6602 |
| 16 | 0.0003 | 0.5 | 64 | 18 | 0.8969 | 0.8157 | 0.7103 | 0.6751 |
| 17 | 0.003 | 0.5 | 128 | 10 | 0.8945 | 0.8214 | 0.7396 | 0.7396 |
| 18 | 0.0003 | 0.1 | 64 | 24 | 0.8932 | 0.8418 | 0.7990 | 0.8356 |
| 19 | 0.0003 | 0.3 | 64 | 34 | 0.8886 | 0.8404 | 0.7878 | 0.8391 |
| 20 | 0.001 | 0.3 | 128 | 10 | 0.8882 | 0.8032 | 0.7220 | 0.7130 |
| 21 | 0.001 | 0.1 | 128 | 10 | 0.8850 | 0.7959 | 0.6874 | 0.6431 |
| 22 | 0.0003 | 0.1 | 32 | 15 | 0.8840 | 0.8196 | 0.7112 | 0.6558 |
| 23 | 0.0003 | 0.3 | 32 | 16 | 0.8835 | 0.8193 | 0.7402 | 0.7289 |
| 24 | 0.0003 | 0.1 | 128 | 28 | 0.8832 | 0.8157 | 0.7436 | 0.7160 |
| 25 | 0.0003 | 0.5 | 32 | 22 | 0.8793 | 0.8245 | 0.7764 | 0.8066 |
| 26 | 0.0003 | 0.5 | 128 | 22 | 0.8773 | 0.7905 | 0.6981 | 0.6472 |
| 27 | 0.0003 | 0.3 | 128 | 22 | 0.8749 | 0.7953 | 0.7408 | 0.7178 |

## Best configuration on validation

- Learning rate: `0.001`
- Dropout: `0.3`
- Batch size: `32`
- Validation AP: `0.9282`

## Interpretation and limitations

The search selects hyperparameters using the same validation split across 27 trials.
Therefore the best validation score is optimistically selected and is not an
independent estimate of generalization. The chosen configuration must be
retrained with at least three seeds and eventually tested on the held-out
test split after model selection is finished.

Reproduce the search from the repository root:

```powershell
python .\scripts\training\grid_search_cnn.py
```
