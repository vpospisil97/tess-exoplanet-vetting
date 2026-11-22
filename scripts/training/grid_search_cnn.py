
"""Grid search for the existing two-view TESS CNN (Model A).

Keeps original model checkpoints untouched. Run from the project root:
    python ./scripts/training/grid_search_cnn.py --max-trials 1
    python ./scripts/training/grid_search_cnn.py
"""

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "full_dataset"
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_cnn.py"
SEARCH_DIR = DATA / "cnn_hparam_search"
DOC_PATH = ROOT / "docs" / "model_a_grid_search.md"

LEARNING_RATES = (0.0003, 0.001, 0.003)
DROPOUTS = (0.1, 0.3, 0.5)
BATCH_SIZES = (32, 64, 128)

SEED = 42
EPOCHS = 35
PATIENCE = 7
WEIGHT_DECAY = 0.0001


def build_configs():
    """Vytvoří úplnou mřížku hyperparametrů CNN."""
    configs = []
    for lr, dropout, batch_size in itertools.product(
        LEARNING_RATES, DROPOUTS, BATCH_SIZES
    ):
        label = (
            f"lr_{str(lr).replace('.', 'p')}"
            f"__drop_{str(dropout).replace('.', 'p')}"
            f"__batch_{batch_size}"
        )
        configs.append({
            "id": label,
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "seed": SEED,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "device": "cuda",
        })
    return configs


def load_trainer():
    """Načte trénovací modul CNN pro izolované běhy."""
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(TRAIN_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "_tess_cnn_grid_trainer", TRAIN_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load train_cnn.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # train_cnn.py was originally written for run.py's legacy
    # root-path compatibility layer. Set those paths explicitly.
    for field in ("ROOT", "DATA", "INPUT", "SPLITS", "OUTPUT", "main"):
        if not hasattr(module, field):
            raise RuntimeError(f"Missing {field} in train_cnn.py")

    module.ROOT = ROOT
    module.DATA = DATA
    module.INPUT = DATA / "full_cnn_views.npz"
    module.SPLITS = DATA / "cnn_split_indices.npz"
    return module


def run_trial(trainer, config, trial_dir):
    """Spustí jednu konfiguraci a oddělí její artefakty."""
    trainer.OUTPUT = trial_dir

    args = [
        str(TRAIN_SCRIPT),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--dropout", str(config["dropout"]),
        "--patience", str(config["patience"]),
        "--seed", str(config["seed"]),
        "--device", config["device"],
    ]

    previous_argv = sys.argv
    try:
        sys.argv = args
        trainer.main()
    finally:
        sys.argv = previous_argv


def read_result(config):
    """Ověří výsledek již dokončené konfigurace."""
    trial_dir = SEARCH_DIR / config["id"]
    config_file = trial_dir / "trial_config.json"
    metrics_file = trial_dir / f"metrics_seed_{SEED}.json"

    if not metrics_file.is_file():
        return None
    if not config_file.is_file():
        raise RuntimeError(f"Missing configuration: {config_file}")

    stored = json.loads(config_file.read_text(encoding="utf-8"))
    if stored != config:
        raise RuntimeError(f"Inconsistent config: {trial_dir}")

    result = json.loads(metrics_file.read_text(encoding="utf-8"))
    if result.get("test_set_used") is not False:
        raise RuntimeError(f"Test split used: {trial_dir}")
    if result.get("seed") != SEED:
        raise RuntimeError(f"Unexpected seed: {trial_dir}")

    hyper = result.get("hyperparameters", {})
    for key in ("lr", "dropout", "batch_size", "epochs", "patience", "weight_decay"):
        if hyper.get(key) != config[key]:
            raise RuntimeError(f"Hyperparameter mismatch ({key}): {trial_dir}")

    metrics = result["validation"]
    return {
        "trial_id": config["id"],
        "lr": config["lr"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "seed": SEED,
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "average_precision": float(metrics["average_precision"]),
        "roc_auc": float(metrics["roc_auc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1": float(metrics["f1"]),
        "training_seconds": float(result["training_seconds"]),
    }


def rank_results(configs):
    """Seřadí výsledky podle validační průměrné přesnosti AP."""
    found = [read_result(cfg) for cfg in configs]
    rows = [r for r in found if r is not None]
    return sorted(
        rows,
        key=lambda r: (-r["average_precision"], -r["roc_auc"], r["trial_id"]),
    )


def write_summary(rows, completed, total):
    """Uloží souhrn grid search do tabulky a dokumentace."""
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SEARCH_DIR / "grid_results.csv"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if completed != total:
        return

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model A - CNN hyperparameter grid search",
        "",
        "## Search design",
        "",
        "- 27 combinations: learning rate x dropout x batch size.",
        f"- Learning rates: {', '.join(map(str, LEARNING_RATES))}.",
        f"- Dropout values: {', '.join(map(str, DROPOUTS))}.",
        f"- Batch sizes: {', '.join(map(str, BATCH_SIZES))}.",
        f"- Fixed seed: {SEED}; epochs: {EPOCHS}; patience: {PATIENCE}; weight decay: {WEIGHT_DECAY}.",
        "- Checkpoint selection and ranking: validation Average Precision (AP).",
        "- The original three CNN seed runs are not overwritten.",
        "- The independent test split was not evaluated.",
        "",
        "## Ranked results",
        "",
        "| Rank | Learning rate | Dropout | Batch | Best epoch | AP | ROC-AUC | Balanced accuracy | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['lr']} | {row['dropout']} | {row['batch_size']} "
            f"| {row['best_epoch']} | {row['average_precision']:.4f} "
            f"| {row['roc_auc']:.4f} | {row['balanced_accuracy']:.4f} "
            f"| {row['f1']:.4f} |"
        )

    best = rows[0]
    lines.extend([
        "",
        "## Best configuration on validation",
        "",
        f"- Learning rate: `{best['lr']}`",
        f"- Dropout: `{best['dropout']}`",
        f"- Batch size: `{best['batch_size']}`",
        f"- Validation AP: `{best['average_precision']:.4f}`",
        "",
        "## Interpretation and limitations",
        "",
        "The search selects hyperparameters using the same validation split across 27 trials.",
        "Therefore the best validation score is optimistically selected and is not an",
        "independent estimate of generalization. The chosen configuration must be",
        "retrained with at least three seeds and eventually tested on the held-out",
        "test split after model selection is finished.",
        "",
        "Reproduce the search from the repository root:",
        "",
        "```powershell",
        "python .\\scripts\\training\\grid_search_cnn.py",
        "```",
        "",
    ])
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Provede reprodukovatelné ladění hyperparametrů CNN."""
    parser = argparse.ArgumentParser(description="Model A - safe CNN grid search")
    parser.add_argument(
        "--max-trials", type=int, default=None,
        help="Run only the first N configurations (for smoke testing)",
    )
    args = parser.parse_args()
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")

    configs = build_configs()
    planned = configs if args.max_trials is None else configs[:args.max_trials]

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check PyTorch and GPU")
    for path in (DATA / "full_cnn_views.npz", DATA / "cnn_split_indices.npz"):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Grid size: {len(configs)}; configurations checked this run: {len(planned)}")
    trainer = None

    for number, config in enumerate(planned, start=1):
        trial_dir = SEARCH_DIR / config["id"]
        previous = read_result(config)
        if previous is not None:
            print(f"[{number}/{len(planned)}] SKIP: {config['id']} (already completed)")
            continue

        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "trial_config.json"
        if config_path.is_file():
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
            if previous_config != config:
                raise RuntimeError(f"Configuration collision: {trial_dir}")
        else:
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        if trainer is None:
            trainer = load_trainer()

        print(f"\n[{number}/{len(planned)}] TRAIN: {config['id']}", flush=True)
        run_trial(trainer, config, trial_dir)
        completed = read_result(config)
        if completed is None:
            raise RuntimeError(f"Training produced no metrics: {trial_dir}")

        rows = rank_results(configs)
        write_summary(rows, len(rows), len(configs))
        print(f"Completed: {len(rows)}/{len(configs)} | AP={completed['average_precision']:.4f}")

    rows = rank_results(configs)
    write_summary(rows, len(rows), len(configs))
    print("\n" + "=" * 70)
    print(f"GRID RESULTS: {len(rows)}/{len(configs)} completed")
    print("=" * 70)
    for rank, row in enumerate(rows[:5], start=1):
        print(
            f"{rank}. lr={row['lr']} dropout={row['dropout']} "
            f"batch={row['batch_size']} AP={row['average_precision']:.4f} "
            f"ROC-AUC={row['roc_auc']:.4f}"
        )
    print("Summary CSV:", SEARCH_DIR / "grid_results.csv")
    if len(rows) == len(configs):
        print("Full Markdown report:", DOC_PATH)
        print("GRID SEARCH COMPLETE")
    else:
        print("Grid incomplete; rerun without --max-trials to finish")
    print("TEST SPLIT NOT USED")


if __name__ == "__main__":
    main()
