# Model B - Corrected Metadata Audit

## Purpose

Inspect eight persistent validation errors
using original TESS signal metadata.

The sample deliberately contains four
confident false positives and four
confident false negatives.

## Verification

- All eight signal UIDs matched.
- TIC identifiers matched for all signals.
- Metadata labels matched the prepared binary labels.
- No held-out test examples were used.

## Selected signals

| error_type     | signal_uid   |    tic_id |   model_score | signal_label   |   period_days |   duration_hours |   depth_ppm |   planet_snr | tess_disposition   | tfopwg_disposition   |
|:---------------|:-------------|----------:|--------------:|:---------------|--------------:|-----------------:|------------:|-------------:|:-------------------|:---------------------|
| false_positive | TOI:1888.01  | 459997997 |        0.8908 | false_positive |        5.5475 |           5.4120 |   3350.0000 |      23.0000 | EB                 | FP                   |
| false_positive | TOI:4279.01  | 407146536 |        0.8867 | false_positive |        2.0977 |           3.3618 |   2503.6857 |      26.4109 | O                  | FP                   |
| false_positive | TOI:2610.01  | 268914377 |        0.8543 | false_positive |        2.1539 |           2.1777 |  96363.0927 |      17.8571 | EB                 | FP                   |
| false_positive | TOI:591.01   | 146261607 |        0.8537 | false_positive |        8.4915 |           6.0350 |  40640.0000 |     230.0000 | EB                 | FP                   |
| false_negative | TOI:6030.01  | 245392284 |        0.0324 | confirmed      |        1.7443 |           2.1928 |   7445.2479 |      16.3654 | KP                 | KP                   |
| false_negative | TOI:2202.01  | 358107516 |        0.0668 | confirmed      |       11.9125 |           3.7521 |  19304.2778 |      90.0533 | CP                 | CP                   |
| false_negative | TOI:1130.01  | 254113311 |        0.1100 | confirmed      |        8.3500 |           1.8360 |  16680.0000 |     129.0000 | CP                 | CP                   |
| false_negative | TOI:1789.01  | 172518755 |        0.1103 | confirmed      |        3.2087 |           1.6390 |   1960.0000 |      53.0000 | CP                 | CP                   |

## Units

- `period_days`: days
- `duration_hours`: hours
- `depth_ppm`: parts per million
- `planet_snr`: signal-to-noise ratio
- `epoch_bjd`: Barycentric Julian Date

## Data quality

- Missing period_days: 0/8
- Missing duration_hours: 0/8
- Missing depth_ppm: 0/8
- Missing planet_snr: 0/8
- Missing epoch_bjd: 0/8

## Interpretation cautions

- Model scores are not calibrated probabilities.
- Deep transits need not originate from planets.
- Labels and dispositions are taken from
  the source metadata and may reflect
  different classification conventions.
- The eight signals were selected because
  the classifier made persistent errors.
- This sample is not representative
  of the complete validation dataset.
- Identifying a physical false-positive
  mechanism requires additional evidence.

## Reproduction

```powershell
python -u .\scripts\analysis\audit_mantis_error_metadata.py
```
