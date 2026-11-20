
"""
MODEL B - MANTIS FEATURE EXTRACTION
===================================

Extract pretrained Mantis-8M embeddings from
the TESS dataset.

Input:
    full_dataset/full_cnn_views.npz
    full_dataset/cnn_split_indices.npz

Output:
    full_dataset/mantis_model_b/features_train.npz
    full_dataset/mantis_model_b/features_validation.npz
    full_dataset/mantis_model_b/extraction_summary.json

Report:
    docs/model_b_feature_extraction.md

Important:
    - Uses pretrained MantisV1.
    - No classifier training.
    - No fine-tuning.
    - No test-set evaluation.
    - Train and validation remain separate.
    - The original dataset is not modified.

Run:
    python -u scripts/training/extract_mantis_features.py
"""

import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset"

INPUT = DATA / "full_cnn_views.npz"

SPLITS = DATA / "cnn_split_indices.npz"

OUTPUT = DATA / "mantis_model_b"

REPORT = (
    ROOT / "docs" / "model_b_feature_extraction.md"
)

CHECKPOINT = "paris-noah/Mantis-8M"

DEVICE = "cuda"

LENGTH = 512

BATCH_SIZE = 16

EXPECTED_DATASET_SIZE = 1918

EXPECTED_SPLITS = {
    "train": 1335,
    "validation": 293,
}

EXPECTED_LABEL_COUNTS = {
    "train": {
        0: 443,
        1: 892,
    },
    "validation": {
        0: 94,
        1: 199,
    },
}


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_view(data):
    """
    Convert a TESS light-curve view into
    Mantis-compatible input.

    Input:
        (N, sequence_length, 2)

    Channel 0:
        Flux

    Channel 1:
        Coverage mask

    The pilot Model B uses the flux channel only.

    Processing:
        1. Select flux
        2. Apply arcsinh(flux / 5)
        3. Linearly resize to 512 samples

    Output:
        (N, 1, 512)
    """

    if data.ndim != 3:
        raise ValueError(
            f"Expected three dimensions, got {data.shape}"
        )

    if data.shape[2] != 2:
        raise ValueError(
            f"Expected two channels, got {data.shape}"
        )

    flux = np.asarray(
        data[:, :, 0],
        dtype=np.float32,
    )

    if not np.isfinite(flux).all():
        raise ValueError(
            "Input flux contains non-finite values"
        )

    transformed = np.arcsinh(
        flux / 5.0
    ).astype(np.float32)

    tensor = torch.from_numpy(
        transformed[:, None, :]
    )

    resized = F.interpolate(
        tensor,
        size=LENGTH,
        mode="linear",
        align_corners=False,
    )

    result = (
        resized.numpy()
        .astype(np.float32, copy=False)
    )

    if not np.isfinite(result).all():
        raise ValueError(
            "Resized input contains non-finite values"
        )

    return result


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_indices(indices, count, name):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if indices.ndim != 1:
        raise ValueError(
            f"{name}: indices must be one-dimensional"
        )

    if indices.size != EXPECTED_SPLITS[name]:
        raise ValueError(
            f"{name}: unexpected split size "
            f"{indices.size}"
        )

    if indices.min() < 0 or indices.max() >= count:
        raise ValueError(
            f"{name}: indices out of range"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            f"{name}: duplicate indices"
        )

    return indices


def load_dataset():
    print("\nLoading dataset...")

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    if not SPLITS.is_file():
        raise FileNotFoundError(SPLITS)

    with np.load(
        INPUT,
        allow_pickle=False,
    ) as data:

        required = (
            "x_local",
            "x_global",
            "y",
            "signal_uid",
            "tic_key",
        )

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"Missing dataset field: {key}"
                )

        x_local = np.asarray(
            data["x_local"],
            dtype=np.float32,
        )

        x_global = np.asarray(
            data["x_global"],
            dtype=np.float32,
        )

        y = np.asarray(
            data["y"],
            dtype=np.int64,
        )

        signal_uid = np.asarray(
            data["signal_uid"]
        ).astype(str)

        tic_key = np.asarray(
            data["tic_key"]
        ).astype(str)

    expected_local = (
        EXPECTED_DATASET_SIZE,
        101,
        2,
    )

    expected_global = (
        EXPECTED_DATASET_SIZE,
        201,
        2,
    )

    if x_local.shape != expected_local:
        raise ValueError(
            f"Unexpected local shape: {x_local.shape}"
        )

    if x_global.shape != expected_global:
        raise ValueError(
            f"Unexpected global shape: {x_global.shape}"
        )

    if y.shape != (EXPECTED_DATASET_SIZE,):
        raise ValueError(
            f"Unexpected label shape: {y.shape}"
        )

    if signal_uid.shape != y.shape:
        raise ValueError(
            "Signal UID shape mismatch"
        )

    if tic_key.shape != y.shape:
        raise ValueError(
            "TIC key shape mismatch"
        )

    if not np.isin(y, (0, 1)).all():
        raise ValueError(
            "Labels must contain only 0 and 1"
        )

    if not np.isfinite(x_local).all():
        raise ValueError(
            "Non-finite local input"
        )

    if not np.isfinite(x_global).all():
        raise ValueError(
            "Non-finite global input"
        )

    with np.load(
        SPLITS,
        allow_pickle=False,
    ) as split:

        for key in (
            "train_idx",
            "validation_idx",
        ):
            if key not in split.files:
                raise KeyError(
                    f"Missing split: {key}"
                )

        train_idx = validate_indices(
            split["train_idx"],
            EXPECTED_DATASET_SIZE,
            "train",
        )

        validation_idx = validate_indices(
            split["validation_idx"],
            EXPECTED_DATASET_SIZE,
            "validation",
        )

    overlap = np.intersect1d(
        train_idx,
        validation_idx,
    )

    if overlap.size:
        raise RuntimeError(
            "Train and validation indices overlap"
        )

    train_tic = set(
        tic_key[train_idx].tolist()
    )

    validation_tic = set(
        tic_key[validation_idx].tolist()
    )

    if train_tic.intersection(validation_tic):
        raise RuntimeError(
            "TIC leakage between train and validation"
        )

    for name, indices in (
        ("train", train_idx),
        ("validation", validation_idx),
    ):
        labels, counts = np.unique(
            y[indices],
            return_counts=True,
        )

        actual = {
            int(label): int(count)
            for label, count in zip(labels, counts)
        }

        if actual != EXPECTED_LABEL_COUNTS[name]:
            raise RuntimeError(
                f"{name}: unexpected label distribution "
                f"{actual}"
            )

    print("Dataset size:", len(y))
    print("Local shape:", x_local.shape)
    print("Global shape:", x_global.shape)
    print("Train signals:", len(train_idx))
    print("Validation signals:", len(validation_idx))
    print("Train TICs:", len(train_tic))
    print("Validation TICs:", len(validation_tic))
    print("TIC leakage check: OK")
    print("Test observations: not selected")

    return {
        "x_local": x_local,
        "x_global": x_global,
        "y": y,
        "signal_uid": signal_uid,
        "tic_key": tic_key,
        "train_idx": train_idx,
        "validation_idx": validation_idx,
    }


# ============================================================
# MANTIS
# ============================================================

def load_mantis():
    print("\nLoading pretrained Mantis...")

    network = MantisV1(
        device=DEVICE,
        return_transf_layer=2,
        output_token="combined",
    )

    network = network.from_pretrained(
        CHECKPOINT
    )

    network.eval()

    trainer = MantisTrainer(
        device=DEVICE,
        network=network,
    )

    parameters = sum(
        p.numel()
        for p in network.parameters()
    )

    print("Checkpoint:", CHECKPOINT)
    print("Parameters:", f"{parameters:,}")

    return trainer, parameters


def extract_features(trainer, inputs, name):
    print(f"\nExtracting {name} embeddings...")

    print("Input shape:", inputs.shape)

    start = time.perf_counter()

    features = np.asarray(
        trainer.transform(
            inputs,
            batch_size=BATCH_SIZE,
        ),
        dtype=np.float32,
    )

    elapsed = time.perf_counter() - start

    expected = (
        inputs.shape[0],
        512,
    )

    if features.shape != expected:
        raise RuntimeError(
            f"{name}: unexpected embedding shape "
            f"{features.shape}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            f"{name}: non-finite embeddings"
        )

    print(
        "Embedding shape:",
        features.shape,
    )

    print(
        "Seconds:",
        f"{elapsed:.2f}",
    )

    print(
        "Mean:",
        f"{float(features.mean()):.6f}",
    )

    print(
        "Std:",
        f"{float(features.std()):.6f}",
    )

    return features, elapsed


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    name,
    indices,
    data,
    trainer,
):
    print()
    print("=" * 70)
    print(f"PROCESSING: {name.upper()}")
    print("=" * 70)

    local = data["x_local"][indices]

    global_view = data["x_global"][indices]

    x_local = prepare_view(local)

    x_global = prepare_view(global_view)

    if x_local.shape != (
        len(indices), 1, LENGTH
    ):
        raise RuntimeError(
            "Unexpected prepared local shape"
        )

    if x_global.shape != (
        len(indices), 1, LENGTH
    ):
        raise RuntimeError(
            "Unexpected prepared global shape"
        )

    z_local, local_seconds = extract_features(
        trainer,
        x_local,
        f"{name} local",
    )

    z_global, global_seconds = extract_features(
        trainer,
        x_global,
        f"{name} global",
    )

    combined = np.concatenate(
        (
            z_local,
            z_global,
        ),
        axis=1,
    )

    if combined.shape != (
        len(indices), 1024
    ):
        raise RuntimeError(
            "Unexpected combined feature shape"
        )

    if not np.isfinite(combined).all():
        raise RuntimeError(
            "Non-finite combined features"
        )

    variation = float(
        combined.std(axis=0).mean()
    )

    if variation <= 0:
        raise RuntimeError(
            "Embeddings have no variation"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT
        / f"features_{name}.npz"
    )

    np.savez_compressed(
        path,
        indices=indices,
        y=data["y"][indices],
        signal_uid=data["signal_uid"][indices],
        tic_key=data["tic_key"][indices],
        z_local=z_local,
        z_global=z_global,
        features=combined,
    )

    print("\nSaved:", path)
    print("Combined shape:", combined.shape)
    print("Mean feature variation:", variation)

    return {
        "samples": int(len(indices)),
        "local_shape": list(z_local.shape),
        "global_shape": list(z_global.shape),
        "combined_shape": list(combined.shape),
        "local_seconds": round(local_seconds, 3),
        "global_seconds": round(global_seconds, 3),
        "feature_variation": variation,
        "file": str(path),
    }


# ============================================================
# REPORT
# ============================================================

def save_report(summary):
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = summary["splits"]["train"]
    validation = summary["splits"]["validation"]

    lines = [
        "# Model B - Pretrained Mantis Feature Extraction",
        "",
        "## 1. Model selection",
        "",
        "A targeted literature review considered",
        "MOMENT and Mantis as pretrained",
        "time-series models.",
        "",
        "Mantis-8M was selected for its",
        "classification-oriented design,",
        "available pretrained weights,",
        "and relatively small size.",
        "",
        "## 2. Pretrained model",
        "",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Parameters: {summary['parameters']:,}",
        f"- Mantis package: `{summary['mantis_version']}`",
        f"- PyTorch: `{summary['torch_version']}`",
        f"- GPU: `{summary['gpu']}`",
        "",
        "## 3. Dataset",
        "",
        f"- Train samples: {train['samples']}",
        f"- Validation samples: {validation['samples']}",
        "- Train/validation split by TIC.",
        "- Train and validation TIC identifiers do not overlap.",
        "- Held-out test signals were not processed.",
        "",
        "## 4. Input preprocessing",
        "",
        "- Start from existing phase-folded TESS views.",
        "- Local input: 101 bins.",
        "- Global input: 201 bins.",
        "- Use flux channel only.",
        "- Apply `arcsinh(flux / 5)`.",
        "- Interpolate each view independently to 512 bins.",
        "- Extract 512-dimensional embeddings per view.",
        "- Concatenate into 1024-dimensional representations.",
        "",
        "## 5. Extracted representations",
        "",
        "| Split | Samples | Local | Global | Combined |",
        "|---|---:|---|---|---|",
        (
            f"| Train | {train['samples']} "
            f"| {train['local_shape']} "
            f"| {train['global_shape']} "
            f"| {train['combined_shape']} |"
        ),
        (
            f"| Validation | {validation['samples']} "
            f"| {validation['local_shape']} "
            f"| {validation['global_shape']} "
            f"| {validation['combined_shape']} |"
        ),
        "",
        "## 6. Methodological notes",
        "",
        "- The pretrained encoder was not fine-tuned.",
        "- No classifier was trained in this step.",
        "- The held-out test set was not evaluated.",
        "- Both views share the same pretrained encoder.",
        "- Linear interpolation may smooth transit morphology.",
        "- Flux-only input does not explicitly include coverage masks.",
        "- Zero-filled missing bins can still encode missingness.",
        "- Suitability for TESS classification requires validation.",
        "",
        "## 7. Next step",
        "",
        "Fit a classification head using train embeddings only.",
        "Evaluate model-selection metrics on validation embeddings.",
        "Repeat key classifier comparisons over three seeds.",
        "",
        "## 8. Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\extract_mantis_features.py",
        "```",
        "",
    ]

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Report saved:", REPORT)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MODEL B - MANTIS FULL FEATURE EXTRACTION")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    print("GPU:", torch.cuda.get_device_name(0))

    np.random.seed(42)
    torch.manual_seed(42)

    data = load_dataset()

    trainer, parameters = load_mantis()

    torch.cuda.reset_peak_memory_stats()

    results = {}

    for name in (
        "train",
        "validation",
    ):
        indices = data[f"{name}_idx"]

        results[name] = process_split(
            name,
            indices,
            data,
            trainer,
        )

    peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    summary = {
        "model": "MantisV1",
        "checkpoint": CHECKPOINT,
        "mantis_version": version("mantis-tsfm"),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": parameters,
        "input_length": LENGTH,
        "batch_size": BATCH_SIZE,
        "preprocessing": "arcsinh(flux / 5), linear interpolation",
        "coverage_mask_used": False,
        "embedding_size_per_view": 512,
        "combined_embedding_size": 1024,
        "splits": results,
        "peak_gpu_memory_mb": round(
            peak_mb,
            2,
        ),
        "test_set_used": False,
        "classifier_trained": False,
    }

    summary_path = (
        OUTPUT / "extraction_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_report(summary)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        print(
            f"{name}:",
            result["combined_shape"],
        )

    print(
        "Peak GPU memory:",
        f"{peak_mb:.2f} MB",
    )

    print("Test data used: NO")
    print("Classifier trained: NO")

    print()
    print("MANTIS FEATURE EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
