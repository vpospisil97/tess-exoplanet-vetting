# TESS - Pretest Integrity Audit

## Dataset

- 1918 labeled binary signals
- Local views: 1918 x 101 x 2
- Global views: 1918 x 201 x 2

## Splits

| Split | Signals | Unique TICs |
|---|---:|---:|
| train | 1335 | 1252 |
| validation | 293 | 268 |
| test | 290 | 269 |

- No overlapping signal indices
- No overlapping TIC identifiers
- All observations assigned exactly once

## Model B

- Training and validation embeddings aligned
- Selected MLP: hidden_128__drop_0p5
- Three selected checkpoints found and metadata verified
- Checkpoint tensor contents not loaded in this audit

## SHA-256 fingerprints

- Views: `ec8b3ef69c3f591b0aa027309ccdbef6e85c9943a838577d031e1c15461f8fb8`
- Splits: `d53c2e68ad021d4a4dd6915da702e488129f92af1a654c5ba007de2723199c54`

## Held-out test

Only split indices and TIC metadata were inspected.
No test predictions or test metrics were calculated.
