
"""
Model B: train a small MLP classification head
on frozen Mantis-8M embeddings.

Run from the project root:

    python -u scripts/training/train_mantis_head.py --max-runs 1

Then:

    python -u scripts/training/train_mantis_head.py

Test examples are never loaded or evaluated.
"""

import argparse
import copy
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CESTY A KONFIGURACE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset"

FEATURE_DIR = DATA / "mantis_model_b"

SPLITS = DATA / "cnn_split_indices.npz"

OUTPUT = FEATURE_DIR / "mlp_head_v1"

REPORT = ROOT / "docs" / "model_b_mlp_validation.md"

SEEDS = (42, 123, 2026)

CONFIG = {
    "head": "mlp_1024_128_1",
    "embedding_source": "paris-noah/Mantis-8M, frozen",
    "hidden_size": 128,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "batch_size": 64,
    "max_epochs": 80,
    "patience": 12,
    "selection_metric": "validation_average_precision",
    "threshold": 0.5,
}

METRICS = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "precision",
    "recall",
)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(name, expected_count, expected_labels):

    """Načte embeddingy a zkontroluje počty vzorků a tříd."""
    path = FEATURE_DIR / f"features_{name}.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as f:

        needed = (
            "indices",
            "y",
            "tic_key",
            "signal_uid",
            "features",
        )

        if any(key not in f.files for key in needed):
            raise ValueError(
                f"Missing feature arrays in {path}"
            )

        x = np.asarray(
            f["features"],
            dtype=np.float32,
        )

        y = np.asarray(
            f["y"],
            dtype=np.int64,
        )

        idx = np.asarray(
            f["indices"],
            dtype=np.int64,
        )

        tic = np.asarray(
            f["tic_key"]
        ).astype(str)

        uid = np.asarray(
            f["signal_uid"]
        ).astype(str)

    if x.shape != (expected_count, 1024):
        raise ValueError(
            f"Unexpected {name} features: {x.shape}"
        )

    if any(
        a.shape != (expected_count,)
        for a in (y, idx, tic, uid)
    ):
        raise ValueError(
            f"Unexpected {name} metadata shapes"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite {name} features"
        )

    if len(np.unique(idx)) != expected_count:
        raise ValueError(
            f"Duplicate {name} indices"
        )

    if any(t == "" for t in tic):
        raise ValueError(
            f"Empty {name} TIC identifier"
        )

    labels, counts = np.unique(
        y,
        return_counts=True,
    )

    observed = {
        int(k): int(v)
        for k, v in zip(labels, counts)
    }

    if observed != expected_labels:
        raise ValueError(
            f"Unexpected {name} labels: {observed}"
        )

    return {
        "x": x,
        "y": y,
        "indices": idx,
        "tic": tic,
        "uid": uid,
    }


def load_data():

    """Načte embeddingy train/validation a zkontroluje jejich návaznost.
    StandardScaler odhaduje parametry pouze na trénovacích příkladech.
    """
    train = load_split(
        "train",
        1335,
        {0: 443, 1: 892},
    )

    val = load_split(
        "validation",
        293,
        {0: 94, 1: 199},
    )

    if not SPLITS.is_file():
        raise FileNotFoundError(SPLITS)

    with np.load(SPLITS, allow_pickle=False) as f:

        if not np.array_equal(
            train["indices"],
            f["train_idx"],
        ):
            raise ValueError(
                "Train indices do not match original split"
            )

        if not np.array_equal(
            val["indices"],
            f["validation_idx"],
        ):
            raise ValueError(
                "Validation indices do not match original split"
            )

    if set(train["tic"]).intersection(val["tic"]):
        raise RuntimeError(
            "TIC leakage between train and validation"
        )

    if np.intersect1d(
        train["indices"],
        val["indices"],
    ).size:
        raise RuntimeError(
            "Index leakage"
        )

    print(
        f"Train: {len(train['y'])} signals / "
        f"{len(set(train['tic']))} TICs"
    )

    print(
        f"Validation: {len(val['y'])} signals / "
        f"{len(set(val['tic']))} TICs"
    )

    print("TIC leakage check: OK")
    print("Test examples: not loaded")

    # Fit normalization on TRAIN only.
    scaler = StandardScaler()

    train["x"] = scaler.fit_transform(
        train["x"]
    ).astype(np.float32)

    val["x"] = scaler.transform(
        val["x"]
    ).astype(np.float32)

    if (
        not np.isfinite(train["x"]).all()
        or not np.isfinite(val["x"]).all()
    ):
        raise RuntimeError(
            "Non-finite normalized embeddings"
        )

    return train, val, scaler


# ============================================================
# KLASIFIKAČNÍ HLAVA
# ============================================================

def make_model():

    """Sestaví trénovanou MLP hlavu nad 1024rozměrnými embeddingy."""
    return nn.Sequential(
        nn.Linear(
            1024,
            CONFIG["hidden_size"],
        ),
        nn.ReLU(),
        nn.Dropout(
            CONFIG["dropout"]
        ),
        nn.Linear(
            CONFIG["hidden_size"],
            1,
        ),
    )


# ============================================================
# METRIKY A PREDIKCE
# ============================================================

def score_metrics(y, probabilities):

    """Vypočítá metriky předpovědí a matici záměn."""
    predictions = (
        probabilities >= CONFIG["threshold"]
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "average_precision": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict(model, x, device):

    """Získá predikovaná skóre klasifikační hlavy bez gradientů."""
    model.eval()

    with torch.inference_mode():

        logits = model(
            torch.from_numpy(x).to(device)
        ).squeeze(-1)

        return (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )


# ============================================================
# ULOŽENÉ BĚHY
# ============================================================

def directory(seed):
    """Určí adresář artefaktů pro konkrétní seed."""
    return OUTPUT / f"seed_{seed}"


def read_completed(seed):

    """Ověří, zda dřívější běh obsahuje všechny artefakty."""
    folder = directory(seed)

    metrics_path = folder / "metrics.json"

    if not metrics_path.exists():
        return None

    for required in (
        "best_head.pt",
        "predictions.csv",
        "history.csv",
    ):
        if not (folder / required).is_file():
            raise RuntimeError(
                f"Incomplete run: {folder}"
            )

    saved = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        saved.get("seed") != seed
        or saved.get("config") != CONFIG
    ):
        raise RuntimeError(
            f"Stored configuration mismatch: {folder}"
        )

    if saved.get("test_set_used") is not False:
        raise RuntimeError(
            "Unexpected test usage"
        )

    return saved


# ============================================================
# TRÉNOVÁNÍ JEDNOHO SEEDU
# ============================================================

def train_seed(seed, train, val, scaler, device):

    """Natrénuje jednu inicializaci klasifikační hlavy na daném seedu.
    Nejlepší checkpoint vybírá výhradně podle validační AP.
    """
    folder = directory(seed)

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = make_model().to(device)

    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    negatives = int(
        (train["y"] == 0).sum()
    )

    positives = int(
        (train["y"] == 1).sum()
    )

    weight_positive = float(
        negatives / positives
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            weight_positive,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    dataset = TensorDataset(
        torch.from_numpy(
            train["x"]
        ),
        torch.from_numpy(
            train["y"].astype(np.float32)
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(
            seed
        ),
        pin_memory=(
            device.type == "cuda"
        ),
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None

    history = []

    start = time.perf_counter()

    print("Head parameters:", param_count)
    print("Positive class weight:", weight_positive)
    print("Training started")

    for epoch in range(
        1,
        CONFIG["max_epochs"] + 1,
    ):

        model.train()

        loss_total = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            ).squeeze(-1)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            loss_total += (
                loss.item() * len(batch_y)
            )

        probabilities = predict(
            model,
            val["x"],
            device,
        )

        ap = float(
            average_precision_score(
                val["y"],
                probabilities,
            )
        )

        roc = float(
            roc_auc_score(
                val["y"],
                probabilities,
            )
        )

        loss_mean = (
            loss_total / len(train["y"])
        )

        improved = (
            ap > best_ap + 1e-6
        )

        if improved:

            best_ap = ap
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        history.append({
            "epoch": epoch,
            "train_loss": loss_mean,
            "val_ap": ap,
            "val_roc_auc": roc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss_mean:.4f} | "
            f"val_AP={ap:.4f} | "
            f"val_ROC={roc:.4f}"
            + (" | BEST" if improved else ""),
            flush=True,
        )

        if (
            epoch - best_epoch
            >= CONFIG["patience"]
        ):

            print("Early stopping")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    model.load_state_dict(
        best_state
    )

    probabilities = predict(
        model,
        val["x"],
        device,
    )

    metrics = score_metrics(
        val["y"],
        probabilities,
    )

    result = {
        "model": "pretrained_mantis_v1_frozen_plus_mlp_head",
        "seed": seed,
        "config": CONFIG,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "head_parameters": param_count,
        "train_seconds": round(
            elapsed,
            3,
        ),
        "validation": metrics,
        "test_set_used": False,
    }

    # Save trained head and training-only scaler.
    torch.save(
        {
            "head_state_dict": best_state,
            "config": CONFIG,
            "seed": seed,
            "scaler_mean": scaler.mean_.astype(
                np.float32
            ),
            "scaler_scale": scaler.scale_.astype(
                np.float32
            ),
        },
        folder / "best_head.pt",
    )

    with (
        folder / "history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0]
            ),
        )

        writer.writeheader()
        writer.writerows(history)

    with (
        folder / "predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            (
                "signal_uid",
                "tic_key",
                "true_label",
                "predicted_probability",
            )
        )

        for uid, tic, label, p in zip(
            val["uid"],
            val["tic"],
            val["y"],
            probabilities,
        ):

            writer.writerow(
                (
                    uid,
                    tic,
                    int(label),
                    float(p),
                )
            )

    (
        folder / "metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("BEST MANTIS HEAD CHECKPOINT")
    print("=" * 70)

    print("Seed:", seed)
    print("Best epoch:", best_epoch)

    for name in METRICS:
        print(
            f"{name:20s}: "
            f"{metrics[name]:.4f}"
        )

    print(
        "Training seconds:",
        f"{elapsed:.2f}",
    )

    print("Saved:", folder)
    print("TEST SET NOT USED")

    return result


# ============================================================
# REPORT
# ============================================================

def write_report(results):

    """Sestaví validační srovnání výsledků klasifikační hlavy."""
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Model B - Frozen Mantis-8M + MLP Head",
        "",
        "## Method",
        "",
        "- Pretrained encoder: `paris-noah/Mantis-8M` (frozen).",
        "- Local and global embeddings concatenated into 1024 features.",
        "- StandardScaler fitted only on training embeddings.",
        "- MLP classifier: 1024 -> 128 -> 1.",
        "- ReLU activation and dropout 0.3.",
        "- Weighted BCEWithLogitsLoss.",
        "- AdamW optimizer.",
        "- Same TIC-based train/validation split as Model A.",
        "- Checkpoint selected by validation Average Precision.",
        "- Three random seeds: 42, 123, 2026.",
        "- Test set not evaluated.",
        "",
        "## Individual runs",
        "",
        "| Seed | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in results:

        v = run["validation"]

        lines.append(
            f"| {run['seed']} "
            f"| {run['best_epoch']} "
            f"| {v['average_precision']:.4f} "
            f"| {v['roc_auc']:.4f} "
            f"| {v['balanced_accuracy']:.4f} "
            f"| {v['f1']:.4f} "
            f"| {run['train_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## Mean +/- sample standard deviation",
        "",
        "| Metric | Mantis + MLP | Tuned CNN reference |",
        "|---|---:|---:|",
    ])

    cnn_reference = {
        "average_precision": "0.9219 +/- 0.0127",
        "roc_auc": "0.8597 +/- 0.0054",
        "balanced_accuracy": "0.7685 +/- 0.0428",
        "f1": "0.8131 +/- 0.0300",
    }

    for metric in METRICS:

        values = [
            r["validation"][metric]
            for r in results
        ]

        lines.append(
            f"| {metric} "
            f"| {statistics.mean(values):.4f} "
            f"+/- {statistics.stdev(values):.4f} "
            f"| {cnn_reference.get(metric, 'N/A')} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is frozen-feature transfer learning, not end-to-end fine-tuning.",
        "- The first MLP-head hyperparameters are provisional.",
        "- Both models use validation-selected checkpoints.",
        "- Three seeds do not represent independent data splits.",
        "- Held-out test data remain unevaluated.",
        "- Interpolation and observational artifacts may affect comparisons.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\train_mantis_head.py",
        "```",
        "",
    ])

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Report:", REPORT)


# ============================================================
# HLAVNÍ BĚH
# ============================================================

def main():

    """Spustí opakované trénování hlavy Mantis pro tři seedy."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Mantis-8M frozen-feature "
            "classification head"
        )
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        choices=(1, 2, 3),
        default=3,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL B - MANTIS + MLP CLASSIFICATION HEAD")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train, val, scaler = load_data()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in SEEDS[:args.max_runs]:

        if read_completed(seed) is not None:

            print(
                f"SKIP seed {seed} "
                "- already completed"
            )

            continue

        print()
        print(f"TRAIN SEED {seed}")

        train_seed(
            seed,
            train,
            val,
            scaler,
            device,
        )

    finished = [
        read_completed(seed)
        for seed in SEEDS
    ]

    finished = [
        r
        for r in finished
        if r is not None
    ]

    print()
    print(
        f"Completed: {len(finished)}/3"
    )

    if len(finished) == 3:

        write_report(finished)

        print()
        print("MEAN +/- SAMPLE SD")

        for metric in (
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "f1",
        ):

            values = [
                r["validation"][metric]
                for r in finished
            ]

            print(
                f"{metric:20s}: "
                f"{statistics.mean(values):.4f} +/- "
                f"{statistics.stdev(values):.4f}"
            )

        print(
            "MODEL B BASELINE COMPLETE"
        )

    else:

        print(
            "Run without --max-runs "
            "to finish remaining seeds."
        )

    print("TEST SET NOT USED")


if __name__ == "__main__":
    main()
