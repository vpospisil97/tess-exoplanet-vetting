# Model B - Pretrained Mantis Feature Extraction

## 1. Model selection

A targeted literature review considered
MOMENT and Mantis as pretrained
time-series models.

Mantis-8M was selected for its
classification-oriented design,
available pretrained weights,
and relatively small size.

## 2. Pretrained model

- Checkpoint: `paris-noah/Mantis-8M`
- Parameters: 8,103,936
- Mantis package: `1.1.0`
- PyTorch: `2.14.1+cu130`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

## 3. Dataset

- Train samples: 1335
- Validation samples: 293
- Train/validation split by TIC.
- Train and validation TIC identifiers do not overlap.
- Held-out test signals were not processed.

## 4. Input preprocessing

- Start from existing phase-folded TESS views.
- Local input: 101 bins.
- Global input: 201 bins.
- Use flux channel only.
- Apply `arcsinh(flux / 5)`.
- Interpolate each view independently to 512 bins.
- Extract 512-dimensional embeddings per view.
- Concatenate into 1024-dimensional representations.

## 5. Extracted representations

| Split | Samples | Local | Global | Combined |
|---|---:|---|---|---|
| Train | 1335 | [1335, 512] | [1335, 512] | [1335, 1024] |
| Validation | 293 | [293, 512] | [293, 512] | [293, 1024] |

## 6. Methodological notes

- The pretrained encoder was not fine-tuned.
- No classifier was trained in this step.
- The held-out test set was not evaluated.
- Both views share the same pretrained encoder.
- Linear interpolation may smooth transit morphology.
- Flux-only input does not explicitly include coverage masks.
- Zero-filled missing bins can still encode missingness.
- Suitability for TESS classification requires validation.

## 7. Next step

Fit a classification head using train embeddings only.
Evaluate model-selection metrics on validation embeddings.
Repeat key classifier comparisons over three seeds.

## 8. Reproduction

```powershell
python -u .\scripts\training\extract_mantis_features.py
```
