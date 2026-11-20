
"""
Repeat the selected CNN configuration with three seeds.

The best hyperparameters are selected from the
completed 27-trial grid search using validation AP.

Original CNN runs and grid-search outputs are preserved.

Usage:
    python scripts/training/repeat_best_cnn.py --max-runs 1
    python scripts/training/repeat_best_cnn.py
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import grid_search_cnn as grid


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset"

GRID_CSV = (
    DATA
    / "cnn_hparam_search"
    / "grid_results.csv"
)

OUTPUT_ROOT = (
    DATA
    / "cnn_tuned_repeats"
)

ORIGINAL_DIR = (
    DATA
    / "cnn_model_a"
)

REPORT = (
    ROOT
    / "docs"
    / "model_a_tuned_validation.md"
)

SEEDS = (42, 123, 2026)

METRICS = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "precision",
    "recall",
)


def select_configuration():
    if not GRID_CSV.is_file():
        raise FileNotFoundError(GRID_CSV)

    with GRID_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    configurations = {
        config["id"]: config
        for config in grid.build_configs()
    }

    if len(rows) != 27:
        raise ValueError(
            f"Expected 27 trials; found {len(rows)}"
        )

    ids = [row["trial_id"] for row in rows]

    if len(set(ids)) != 27:
        raise ValueError("Duplicate trial IDs")

    if set(ids) != set(configurations):
        raise ValueError(
            "Grid results do not match planned configs"
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["average_precision"]),
            -float(row["roc_auc"]),
            row["trial_id"],
        ),
    )

    winner = ordered[0]
    config = configurations[winner["trial_id"]]

    return config, winner


def read_run(directory, config, seed):
    path = directory / f"metrics_seed_{seed}.json"

    if not path.is_file():
        return None

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if result["seed"] != seed:
        raise ValueError(
            f"Seed mismatch in {path}"
        )

    if result["test_set_used"] is not False:
        raise ValueError(
            f"Test data were used: {path}"
        )

    if result["model"] != "two_view_1d_cnn":
        raise ValueError(
            f"Unexpected model: {path}"
        )

    for key in (
        "lr",
        "dropout",
        "batch_size",
        "epochs",
        "patience",
        "weight_decay",
    ):
        actual = result["hyperparameters"][key]
        expected = config[key]

        if actual != expected:
            raise ValueError(
                f"Wrong {key} in {path}: "
                f"{actual} != {expected}"
            )

    checkpoint = (
        directory / f"cnn_best_seed_{seed}.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    return result


def read_original(seed):
    path = (
        ORIGINAL_DIR
        / f"metrics_seed_{seed}.json"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if result["seed"] != seed:
        raise ValueError(
            f"Unexpected seed in {path}"
        )

    if result["test_set_used"] is not False:
        raise ValueError(
            "Original model used test data"
        )

    return result


def summary_stats(results, metric):
    values = [
        float(result["validation"][metric])
        for result in results
    ]

    return (
        statistics.mean(values),
        statistics.stdev(values),
    )


def write_report(
    directory,
    config,
    winner,
    tuned_results,
):
    original_results = [
        read_original(seed)
        for seed in SEEDS
    ]

    lines = [
        "# Model A - Tuned CNN Validation",
        "",
        "## 1. Hyperparameter selection",
        "",
        "The configuration was selected from "
        "27 grid-search trials using validation "
        "Average Precision.",
        "",
        f"- Selected trial: `{config['id']}`",
        f"- Learning rate: `{config['lr']}`",
        f"- Dropout: `{config['dropout']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Weight decay: `{config['weight_decay']}`",
        f"- Maximum epochs: `{config['epochs']}`",
        f"- Patience: `{config['patience']}`",
        f"- Grid-search AP: "
        f"`{float(winner['average_precision']):.4f}`",
        "",
        "## 2. Repeated training",
        "",
        "The selected configuration was retrained "
        "using three different random seeds.",
        "",
        "| Seed | Best epoch | AP | ROC-AUC | "
        "Balanced accuracy | F1 | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in tuned_results:
        metrics = result["validation"]

        lines.append(
            f"| {result['seed']} "
            f"| {result['best_epoch']} "
            f"| {metrics['average_precision']:.4f} "
            f"| {metrics['roc_auc']:.4f} "
            f"| {metrics['balanced_accuracy']:.4f} "
            f"| {metrics['f1']:.4f} "
            f"| {result['training_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Comparison with original CNN",
        "",
        "Both configurations use the same "
        "TIC-based training and validation splits.",
        "",
        "| Metric | Original mean +/- SD | "
        "Tuned mean +/- SD |",
        "|---|---:|---:|",
    ])

    for metric in METRICS:
        old_mean, old_std = summary_stats(
            original_results,
            metric,
        )

        new_mean, new_std = summary_stats(
            tuned_results,
            metric,
        )

        lines.append(
            f"| {metric} "
            f"| {old_mean:.4f} +/- {old_std:.4f} "
            f"| {new_mean:.4f} +/- {new_std:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Interpretation and limitations",
        "",
        "- Average Precision is the primary "
        "model-selection metric.",
        "- Both architectures are the same; "
        "only training hyperparameters differ.",
        "- The held-out test split was not evaluated.",
        "- The best configuration was chosen using "
        "the validation set.",
        "- Seed 42 also participated in the original "
        "hyperparameter search.",
        "- Repeated validation results measure "
        "initialization sensitivity, not independent "
        "generalization performance.",
        "- Final performance must be evaluated on "
        "the held-out test data after model selection.",
        "- Potential observational or processing "
        "artifacts still require investigation.",
        "",
        "## 5. Reproduction",
        "",
        "Run from the repository root:",
        "",
        "```powershell",
        "python .\\scripts\\training\\grid_search_cnn.py",
        "python .\\scripts\\training\\repeat_best_cnn.py",
        "```",
        "",
        f"Local experiment directory: `{directory.relative_to(ROOT).as_posix()}`",
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

    print("\nReport:", REPORT)


def main():
    parser = argparse.ArgumentParser(
        description="Repeat best TESS CNN configuration"
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only first N seeds",
    )

    args = parser.parse_args()

    if args.max_runs is not None:
        if not 1 <= args.max_runs <= len(SEEDS):
            parser.error(
                "--max-runs must be between 1 and 3"
            )

    config, winner = select_configuration()

    output_dir = OUTPUT_ROOT / config["id"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "selected_trial": config["id"],
        "configuration": {
            key: value
            for key, value in config.items()
            if key != "seed"
        },
        "selection_metric": "validation AP",
        "grid_search_seed": 42,
    }

    metadata_path = (
        output_dir / "experiment_config.json"
    )

    if metadata_path.is_file():
        stored = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if stored != metadata:
            raise ValueError(
                "Experiment configuration mismatch"
            )

    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    print("=" * 70)
    print("MODEL A - REPEAT BEST CONFIGURATION")
    print("=" * 70)

    print("\nSelected:", config["id"])
    print("Learning rate:", config["lr"])
    print("Dropout:", config["dropout"])
    print("Batch size:", config["batch_size"])
    print("Output directory:", output_dir)

    chosen_seeds = (
        SEEDS
        if args.max_runs is None
        else SEEDS[:args.max_runs]
    )

    trainer = None

    for seed in chosen_seeds:
        run_config = dict(config)
        run_config["seed"] = seed

        existing = read_run(
            output_dir,
            run_config,
            seed,
        )

        if existing is not None:
            print(
                f"\nSeed {seed}: already completed, skipping"
            )
            continue

        if trainer is None:
            trainer = grid.load_trainer()

        print(
            f"\n========== TRAIN SEED {seed} =========="
        )

        grid.run_trial(
            trainer,
            run_config,
            output_dir,
        )

        result = read_run(
            output_dir,
            run_config,
            seed,
        )

        if result is None:
            raise RuntimeError(
                f"Missing metrics for seed {seed}"
            )

    results = []

    for seed in SEEDS:
        run_config = dict(config)
        run_config["seed"] = seed

        result = read_run(
            output_dir,
            run_config,
            seed,
        )

        if result is not None:
            results.append(result)

    print("\n" + "=" * 70)
    print(
        f"COMPLETED: {len(results)}/{len(SEEDS)}"
    )
    print("=" * 70)

    for result in results:
        metrics = result["validation"]

        print(
            f"Seed {result['seed']}: "
            f"AP={metrics['average_precision']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    if len(results) == len(SEEDS):
        print("\nMEAN +/- SAMPLE SD")

        for metric in METRICS:
            mean, std = summary_stats(
                results,
                metric,
            )

            print(
                f"{metric:20s}: "
                f"{mean:.4f} +/- {std:.4f}"
            )

        write_report(
            output_dir,
            config,
            winner,
            results,
        )

        print("\nALL THREE SEEDS COMPLETED")

    else:
        print(
            "\nPartial run complete. "
            "Run again without --max-runs."
        )

    print("TEST SPLIT NOT USED")
    print("DONE")


if __name__ == "__main__":
    main()
