# TESS held-out test: final frozen models

Exactly one held-out test pass, TIC-disjoint from training/validation.
Selected with validation AP; positive class = confirmed; threshold = 0.5.

## Performance (mean ± sample SD over three seeds)

| Model | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistic | 0.7981 | 0.6979 | 0.6541 | 0.6189 |
| CNN | 0.8851 ± 0.0049 | 0.8225 ± 0.0014 | 0.7308 ± 0.0107 | 0.7772 ± 0.0679 |
| Mantis | 0.9009 ± 0.0024 | 0.8434 ± 0.0018 | 0.7539 ± 0.0144 | 0.8119 ± 0.0106 |

## Per-run results

| Model | Seed | AP | ROC-AUC | Balanced acc. | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | baseline | 0.7981 | 0.6979 | 0.6541 | 0.6189 | 78 | 17 | 100 | 95 |
| CNN | 42 | 0.8837 | 0.8241 | 0.7382 | 0.7727 | 74 | 21 | 59 | 136 |
| CNN | 123 | 0.8809 | 0.8221 | 0.7358 | 0.8473 | 56 | 39 | 23 | 172 |
| CNN | 2026 | 0.8905 | 0.8214 | 0.7185 | 0.7117 | 80 | 15 | 79 | 116 |
| Mantis | 42 | 0.8984 | 0.8423 | 0.7374 | 0.8000 | 68 | 27 | 47 | 148 |
| Mantis | 123 | 0.9030 | 0.8455 | 0.7607 | 0.8204 | 70 | 25 | 42 | 153 |
| Mantis | 2026 | 0.9014 | 0.8424 | 0.7636 | 0.8152 | 72 | 23 | 45 | 150 |

## Runtime

Model load time excluded. Mantis encoder extraction and classifier-head inference are measured separately. CNN timing includes device transfers.

- Logistic prediction: 0.002 s
- CNN seed 42: 0.247 s
- CNN seed 123: 0.018 s
- CNN seed 2026: 0.006 s
- Mantis encoder extraction, 2 views: 0.503 s
- Mantis head seed 42: 0.002 s
- Mantis head seed 123: 0.002 s
- Mantis head seed 2026: 0.002 s

## Protocol and limitations

- Test signals: 290, confirmed: 195, false positives: 95
- Device: cuda; torch: 2.14.1+cu130; numpy: 2.2.6
- Checkpoints and preprocessing were frozen after validation.
- No hyperparameters were chosen from test performance.
- Seeds share one fixed TIC-grouped split; SD reflects initialization only.
- A single test split provides limited precision for generalization claims.
- Mantis embeddings were generated with the frozen pretrained encoder.
