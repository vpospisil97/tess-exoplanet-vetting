# TESS - Data Selection Lineage

Audit of existing experiment artifacts.
No FITS data, models or test metrics were regenerated.

## Signal selection

| Stage | Signals | Confirmed | False positive | Removed |
|---|---:|---:|---:|---:|
| Katalogove oznacene signaly | 1978 | 1312 | 666 | 0 |
| Pokryti tranzitu: coverage_pass | 1932 | 1292 | 640 | 46 |
| Delka tranzitu / perioda < 0.25 | 1931 | 1292 | 639 | 1 |
| Uspesne pripraveny pohled | 1918 | 1286 | 632 | 13 |

## Final exclusions

- Original binary signals: 1978
- Final prepared signals: 1918
- Excluded signals: 60

## Preparation rejection reasons

- `insufficient_bins: local=0.00, global=0.96, center=0`: 1
- `insufficient_bins: local=0.03, global=1.00, center=0`: 1
- `insufficient_bins: local=0.24, global=1.00, center=5`: 1
- `insufficient_bins: local=0.26, global=1.00, center=5`: 1
- `insufficient_bins: local=1.00, global=0.03, center=21`: 2
- `insufficient_bins: local=1.00, global=0.04, center=21`: 2
- `insufficient_bins: local=1.00, global=0.05, center=21`: 2
- `insufficient_bins: local=1.00, global=0.06, center=21`: 1
- `insufficient_bins: local=1.00, global=0.08, center=21`: 1
- `insufficient_bins: local=1.00, global=0.20, center=21`: 1

## Integrity

- Final NPZ identifiers match QC rows with status `ok`.
- Binary NPZ labels match original catalogue metadata.
- Preliminary-ready binary signals: 1939
- Final signals absent from preliminary-ready subset: 0
- Held-out test predictions were not regenerated.
