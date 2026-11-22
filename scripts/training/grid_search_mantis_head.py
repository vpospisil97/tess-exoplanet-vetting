
"""
MODEL B - MANTIS CLASSIFICATION HEAD GRID SEARCH
================================================

Systematic hyperparameter search for a classifier
trained on frozen Mantis-8M embeddings.

Configurations:
    Hidden size: 64, 128, 256
    Dropout: 0.1, 0.3, 0.5

Total: 9 configurations, seed 42.

The existing train_mantis_head.py is reused.

The original Mantis model and its checkpoints
remain unchanged.

Run:
    python -u scripts/training/grid_search_mantis_head.py --max-trials 1

Then:
    python -u scripts/training/grid_search_mantis_head.py

Test data is NEVER loaded.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

import train_mantis_head as head


# ============================================================
# CESTY
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    ROOT
    / "full_dataset"
    / "mantis_model_b"
)

OUTPUT = (
    FEATURE_DIR
    / "head_grid_v1"
)

REPORT = (
    ROOT
    / "docs"
    / "model_b_head_grid_search.md"
)


# ============================================================
# NASTAVENÍ EXPERIMENTU
# ============================================================

SEED = 42

HIDDEN_SIZES = (
    64,
    128,
    256,
)

DROPOUTS = (
    0.1,
    0.3,
    0.5,
)

GRID = [
    (hidden, dropout)
    for hidden in HIDDEN_SIZES
    for dropout in DROPOUTS
]

BASE_CONFIG = dict(head.CONFIG)

CSV_FIELDS = (
    "trial",
    "hidden_size",
    "dropout",
    "seed",
    "best_epoch",
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "head_parameters",
    "train_seconds",
)


# ============================================================
# VERIFY INPUT FILES
# ============================================================

def file_sha256(path):

    """Vypočítá kontrolní otisk souboru s příznaky."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:

        while True:

            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def get_feature_hashes():

    """Ověří nezměněné vstupní embeddingy mezi konfiguracemi."""
    result = {}

    for name in (
        "train",
        "validation",
    ):

        path = (
            FEATURE_DIR
            / f"features_{name}.npz"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        result[name] = file_sha256(
            path
        )

    return result


# ============================================================
# KONFIGURACE
# ============================================================

def trial_name(hidden, dropout):

    """Sestaví jednoznačný název konfigurace MLP hlavy."""
    text = str(dropout).replace(
        ".",
        "p",
    )

    return (
        f"hidden_{hidden}"
        f"__drop_{text}"
    )


def trial_directory(hidden, dropout):

    """Určí výstupní adresář zvolené konfigurace."""
    return (
        OUTPUT
        / trial_name(hidden, dropout)
    )


def make_config(
    hidden,
    dropout,
    feature_hashes,
):

    """Sestaví nastavení klasifikační hlavy pro grid search."""
    config = dict(BASE_CONFIG)

    config["hidden_size"] = hidden

    config["dropout"] = dropout

    config["experiment"] = (
        "mantis_head_grid_v1"
    )

    config["trial_name"] = trial_name(
        hidden,
        dropout,
    )

    config["feature_hashes"] = dict(
        feature_hashes
    )

    return config


def activate_trial(
    hidden,
    dropout,
    feature_hashes,
):

    """Aktivuje izolovanou konfiguraci v trénovacím modulu."""
    head.CONFIG = make_config(
        hidden,
        dropout,
        feature_hashes,
    )

    head.OUTPUT = trial_directory(
        hidden,
        dropout,
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_trial(
    hidden,
    dropout,
    feature_hashes,
    train,
    validation,
    scaler,
    device,
    position,
    total,
):

    """Natrénuje a vyhodnotí jednu konfiguraci na validaci."""
    activate_trial(
        hidden,
        dropout,
        feature_hashes,
    )

    name = trial_name(
        hidden,
        dropout,
    )

    print()
    print("=" * 72)

    print(
        f"[{position}/{total}] "
        f"{name}"
    )

    print("=" * 72)

    existing = head.read_completed(
        SEED
    )

    if existing is not None:

        print(
            "SKIP: already completed"
        )

        result = existing

    else:

        print(
            "Hidden size:",
            hidden,
        )

        print(
            "Dropout:",
            dropout,
        )

        print(
            "Random seed:",
            SEED,
        )

        result = head.train_seed(
            SEED,
            train,
            validation,
            scaler,
            device,
        )

    if result.get("test_set_used") is not False:

        raise RuntimeError(
            "Test data were used unexpectedly"
        )

    if result.get("config") != head.CONFIG:

        raise RuntimeError(
            "Configuration mismatch"
        )

    metrics = result["validation"]

    print()
    print(
        "Trial Average Precision:",
        f"{metrics['average_precision']:.4f}"
    )

    print(
        "Trial ROC-AUC:",
        f"{metrics['roc_auc']:.4f}"
    )


# ============================================================
# LOAD ONE COMPLETED RESULT
# ============================================================

def read_trial(
    hidden,
    dropout,
    feature_hashes,
):

    """Načte existující výsledek a ověří jeho konfiguraci."""
    activate_trial(
        hidden,
        dropout,
        feature_hashes,
    )

    result = head.read_completed(
        SEED
    )

    if result is None:
        return None

    metrics = result["validation"]

    return {
        "trial": trial_name(
            hidden,
            dropout,
        ),
        "hidden_size": hidden,
        "dropout": dropout,
        "seed": SEED,
        "best_epoch": int(
            result["best_epoch"]
        ),
        "average_precision": float(
            metrics["average_precision"]
        ),
        "roc_auc": float(
            metrics["roc_auc"]
        ),
        "balanced_accuracy": float(
            metrics["balanced_accuracy"]
        ),
        "f1": float(
            metrics["f1"]
        ),
        "accuracy": float(
            metrics["accuracy"]
        ),
        "precision": float(
            metrics["precision"]
        ),
        "recall": float(
            metrics["recall"]
        ),
        "head_parameters": int(
            result["head_parameters"]
        ),
        "train_seconds": float(
            result["train_seconds"]
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_csv(rows):

    """Uloží výsledky zkoušených konfigurací do CSV."""
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT
        / "grid_results.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(rows)

    return path


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    rows,
    feature_hashes,
):

    """Vytvoří srovnávací zprávu grid search hlavy Mantis."""
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["average_precision"],
            -row["roc_auc"],
            row["trial"],
        ),
    )

    winner = ranked[0]

    config = make_config(
        winner["hidden_size"],
        winner["dropout"],
        feature_hashes,
    )

    selection = {
        "selection_metric": (
            "validation_average_precision"
        ),
        "selection_seed": SEED,
        "selected_trial": winner["trial"],
        "best_validation_ap": winner[
            "average_precision"
        ],
        "config": config,
        "test_set_used": False,
    }

    selection_path = (
        OUTPUT
        / "best_head_config.json"
    )

    selection_path.write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Model B - Mantis Head Grid Search",
        "",
        "## Research objective",
        "",
        "Systematic comparison of hidden-layer "
        "size and dropout for a classifier "
        "trained on frozen Mantis-8M features.",
        "",
        "## Experimental conditions",
        "",
        "- Pretrained encoder: Mantis-8M.",
        "- Encoder weights remain frozen.",
        "- Input features: 1024 dimensions.",
        "- Same TIC-based training/validation split.",
        "- Hidden sizes: 64, 128, 256.",
        "- Dropout: 0.1, 0.3, 0.5.",
        "- Learning rate: 0.001.",
        "- Weight decay: 0.001.",
        "- Batch size: 64.",
        "- Maximum epochs: 80.",
        "- Early stopping patience: 12.",
        "- Grid-search seed: 42.",
        "- Checkpoint selected by validation AP.",
        "- Test set not evaluated.",
        "",
        "## Ranked results",
        "",
        "| Rank | Hidden | Dropout | AP | "
        "ROC-AUC | Balanced accuracy | F1 | Best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(
        ranked,
        start=1,
    ):

        lines.append(
            f"| {rank} "
            f"| {row['hidden_size']} "
            f"| {row['dropout']:.1f} "
            f"| {row['average_precision']:.4f} "
            f"| {row['roc_auc']:.4f} "
            f"| {row['balanced_accuracy']:.4f} "
            f"| {row['f1']:.4f} "
            f"| {row['best_epoch']} |"
        )

    lines.extend([
        "",
        "## Selected configuration",
        "",
        f"- Trial: `{winner['trial']}`",
        f"- Hidden size: {winner['hidden_size']}",
        f"- Dropout: {winner['dropout']:.1f}",
        f"- Validation AP: {winner['average_precision']:.4f}",
        "",
        "## Limitations",
        "",
        "- All trials use the same validation split.",
        "- The selected AP is optimistic because "
        "validation was used for model selection.",
        "- One seed is used for hyperparameter search.",
        "- The selected configuration should "
        "be evaluated across three seeds.",
        "- The held-out test set remains unused.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\grid_search_mantis_head.py",
        "```",
        "",
    ])

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BEST GRID-SEARCH CONFIGURATION")
    print("=" * 72)

    print(
        "Trial:",
        winner["trial"],
    )

    print(
        "Average Precision:",
        f"{winner['average_precision']:.4f}",
    )

    print(
        "ROC-AUC:",
        f"{winner['roc_auc']:.4f}",
    )

    print(
        "Best config:",
        selection_path,
    )

    print(
        "Report:",
        REPORT,
    )


# ============================================================
# HLAVNÍ BĚH
# ============================================================

def main():

    """Spustí systematické hledání šířky a regularizace MLP."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid search for Mantis "
            "classification head"
        )
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Run first N of 9 configurations",
    )

    args = parser.parse_args()

    if args.max_trials is not None:

        if not 1 <= args.max_trials <= 9:

            parser.error(
                "--max-trials must be 1..9"
            )

    print("=" * 72)
    print("MODEL B - MANTIS HEAD GRID SEARCH")
    print("=" * 72)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Total configurations:",
        len(GRID),
    )

    selected = (
        GRID
        if args.max_trials is None
        else GRID[:args.max_trials]
    )

    print(
        "Configurations in this run:",
        len(selected),
    )

    print(
        "Search seed:",
        SEED,
    )

    print()
    print("Checking Mantis feature files...")

    feature_hashes = get_feature_hashes()

    print(
        "Loading training and validation features..."
    )

    train, validation, scaler = (
        head.load_data()
    )

    device = torch.device(
        "cuda"
    )

    for position, (hidden, dropout) in enumerate(
        selected,
        start=1,
    ):

        run_trial(
            hidden,
            dropout,
            feature_hashes,
            train,
            validation,
            scaler,
            device,
            position,
            len(selected),
        )

    rows = []

    for hidden, dropout in GRID:

        result = read_trial(
            hidden,
            dropout,
            feature_hashes,
        )

        if result is not None:
            rows.append(result)

    path = save_csv(
        rows
    )

    print()
    print("=" * 72)
    print("GRID SEARCH SUMMARY")
    print("=" * 72)

    print(
        f"Completed: {len(rows)}/{len(GRID)}"
    )

    print(
        "CSV:",
        path,
    )

    if len(rows) == len(GRID):

        generate_report(
            rows,
            feature_hashes,
        )

        print()
        print(
            "GRID SEARCH COMPLETE"
        )

    else:

        print(
            "Grid search incomplete. "
            "Run again without --max-trials."
        )

    print(
        "TEST SET NOT USED"
    )


if __name__ == "__main__":
    main()
