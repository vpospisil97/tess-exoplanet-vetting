# Model B - Mantis Head Grid Search

## Research objective

Systematic comparison of hidden-layer size and dropout for a classifier trained on frozen Mantis-8M features.

## Experimental conditions

- Pretrained encoder: Mantis-8M.
- Encoder weights remain frozen.
- Input features: 1024 dimensions.
- Same TIC-based training/validation split.
- Hidden sizes: 64, 128, 256.
- Dropout: 0.1, 0.3, 0.5.
- Learning rate: 0.001.
- Weight decay: 0.001.
- Batch size: 64.
- Maximum epochs: 80.
- Early stopping patience: 12.
- Grid-search seed: 42.
- Checkpoint selected by validation AP.
- Test set not evaluated.

## Ranked results

| Rank | Hidden | Dropout | AP | ROC-AUC | Balanced accuracy | F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.5 | 0.9414 | 0.8823 | 0.7984 | 0.8435 | 3 |
| 2 | 64 | 0.3 | 0.9408 | 0.8802 | 0.7934 | 0.8373 | 3 |
| 3 | 64 | 0.5 | 0.9403 | 0.8797 | 0.7950 | 0.8519 | 7 |
| 4 | 256 | 0.5 | 0.9382 | 0.8764 | 0.7777 | 0.8267 | 3 |
| 5 | 64 | 0.1 | 0.9382 | 0.8752 | 0.7825 | 0.8368 | 3 |
| 6 | 128 | 0.3 | 0.9380 | 0.8750 | 0.7906 | 0.8382 | 3 |
| 7 | 128 | 0.1 | 0.9367 | 0.8737 | 0.7755 | 0.8194 | 3 |
| 8 | 256 | 0.1 | 0.9345 | 0.8698 | 0.7699 | 0.8213 | 3 |
| 9 | 256 | 0.3 | 0.9329 | 0.8672 | 0.7727 | 0.8204 | 3 |

## Selected configuration

- Trial: `hidden_128__drop_0p5`
- Hidden size: 128
- Dropout: 0.5
- Validation AP: 0.9414

## Limitations

- All trials use the same validation split.
- The selected AP is optimistic because validation was used for model selection.
- One seed is used for hyperparameter search.
- The selected configuration should be evaluated across three seeds.
- The held-out test set remains unused.

## Reproduction

```powershell
python -u .\scripts\training\grid_search_mantis_head.py
```
