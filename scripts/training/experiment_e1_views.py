"""
E1: Local versus global TESS light-curve views.

Reuses the existing train_cnn.py and grid_search_cnn.py
without modifying their source code.

Run from the project root:

    python -u scripts/training/experiment_e1_views.py --max-trials 1

    python -u scripts/training/experiment_e1_views.py
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import grid_search_cnn as grid


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "full_dataset"
    / "cnn_e1_views"
)

REPORT = (
    ROOT
    / "docs"
    / "model_a_experiment_e1_views.md"
)

SEEDS = (42, 123, 2026)

MODES = (
    "both",
    "local_only",
    "global_only",
)

METRICS = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "accuracy",
)

FIELDS = (
    "mode",
    "seed",
    "best_epoch",
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "training_seconds",
    "trainable_parameters",
)


def configuration(mode, seed):
    return {
        "id": f"e1_{mode}_seed_{seed}",
        "mode": mode,
        "seed": seed,
        "epochs": 35,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.3,
        "patience": 7,
        "device": "cuda",
    }


def trial_directory(mode, seed):
    return (
        OUTPUT
        / mode
        / f"seed_{seed}"
    )


def get_result(mode, seed):
    directory = trial_directory(mode, seed)

    config_path = (
        directory / "trial_config.json"
    )

    metrics_path = (
        directory / f"metrics_seed_{seed}.json"
    )

    checkpoint_path = (
        directory / f"cnn_best_seed_{seed}.pt"
    )

    if not metrics_path.is_file():
        return None

    if (
        not config_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise RuntimeError(
            f"Incomplete experiment output: {directory}"
        )

    expected = configuration(mode, seed)

    stored = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if stored != expected:
        raise RuntimeError(
            f"Experiment settings changed: {directory}"
        )

    result = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    if result.get("test_set_used") is not False:
        raise RuntimeError(
            f"Test data used unexpectedly: {directory}"
        )

    if result.get("seed") != seed:
        raise RuntimeError(
            f"Wrong seed: {directory}"
        )

    if result.get("model") != "two_view_1d_cnn":
        raise RuntimeError(
            f"Unexpected model: {directory}"
        )

    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "patience",
        "device",
    ):
        actual = result["hyperparameters"].get(key)

        if actual != expected[key]:
            raise RuntimeError(
                f"Hyperparameter mismatch ({key}): {directory}"
            )

    m = result["validation"]

    row = {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        **{
            metric: float(m[metric])
            for metric in METRICS
        },
        "training_seconds": float(
            result["training_seconds"]
        ),
        "trainable_parameters": int(
            result["trainable_parameters"]
        ),
    }

    if row["trainable_parameters"] != 79937:
        raise RuntimeError(
            f"Unexpected model capacity: {directory}"
        )

    return row


def make_model_class(original_class, mode):
    """
    Preserve the same network architecture.

    Disabled views have both channels replaced
    by zeros: flux and coverage mask.

    The same ablation is applied during
    training and validation.
    """

    class ViewAblationCNN(original_class):

        def forward(self, local, global_view):

            if mode == "local_only":
                global_view = torch.zeros_like(
                    global_view
                )

            elif mode == "global_only":
                local = torch.zeros_like(
                    local
                )

            return super().forward(
                local,
                global_view,
            )

    return ViewAblationCNN


def write_results(rows):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT / "e1_results.csv"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def average_and_sd(rows, mode, metric):
    values = [
        row[metric]
        for row in rows
        if row["mode"] == mode
    ]

    if len(values) != len(SEEDS):
        raise ValueError(
            f"Missing seeds for {mode}"
        )

    return (
        statistics.mean(values),
        statistics.stdev(values),
    )


def generate_report(rows):
    lines = [
        "# Experiment E1 - Local vs Global Transit Views",
        "",
        "## 1. Research question",
        "",
        "Does the two-view TESS CNN benefit from "
        "information in both the local transit "
        "window and the global phase-folded "
        "light curve?",
        "",
        "**Hypothesis:** using both views gives "
        "higher validation Average Precision "
        "than either view alone.",
        "",
        "## 2. Controlled setup",
        "",
        "- Same TIC-based train, validation, "
        "and test split.",
        "- Same labels and preprocessing.",
        "- Same CNN architecture: "
        "79,937 trainable parameters.",
        "- Learning rate: 0.001.",
        "- Dropout: 0.3.",
        "- Batch size: 32.",
        "- Weight decay: 0.0001.",
        "- Maximum epochs: 35.",
        "- Early stopping patience: 7.",
        "- Seeds: 42, 123, 2026.",
        "- Checkpoint selection: validation AP.",
        "- Classification threshold: 0.5.",
        "- Test set was not evaluated.",
        "",
        "### Input variants",
        "",
        "- Both: original local and global views.",
        "- Local only: global flux and coverage "
        "mask replaced by zeros.",
        "- Global only: local flux and coverage "
        "mask replaced by zeros.",
        "",
        "Disabled inputs are zeroed during "
        "both training and validation.",
        "",
        "## 3. Individual results",
        "",
        "| View | Seed | Epoch | AP | ROC-AUC | "
        "Balanced accuracy | F1 | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['mode']} "
            f"| {row['seed']} "
            f"| {row['best_epoch']} "
            f"| {row['average_precision']:.4f} "
            f"| {row['roc_auc']:.4f} "
            f"| {row['balanced_accuracy']:.4f} "
            f"| {row['f1']:.4f} "
            f"| {row['training_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## 4. Mean and sample standard deviation",
        "",
        "| Metric | Both | Local only | Global only |",
        "|---|---:|---:|---:|",
    ])

    for metric in METRICS:
        values = []

        for mode in MODES:
            mean, sd = average_and_sd(
                rows,
                mode,
                metric,
            )

            values.append(
                f"{mean:.4f} +/- {sd:.4f}"
            )

        lines.append(
            f"| {metric} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend([
        "",
        "## 5. Paired AP differences",
        "",
        "| Seed | Both - Local | Both - Global |",
        "|---:|---:|---:|",
    ])

    for seed in SEEDS:
        by_mode = {
            row["mode"]: row
            for row in rows
            if row["seed"] == seed
        }

        both = by_mode["both"][
            "average_precision"
        ]

        local = by_mode["local_only"][
            "average_precision"
        ]

        global_view = by_mode["global_only"][
            "average_precision"
        ]

        lines.append(
            f"| {seed} "
            f"| {both - local:+.4f} "
            f"| {both - global_view:+.4f} |"
        )

    lines.extend([
        "",
        "## 6. Limitations",
        "",
        "- This is an input-information ablation.",
        "- All CNN branches remain present, but "
        "one branch receives constant zero input "
        "in single-view variants.",
        "- Zeroed inputs are an artificial "
        "experimental condition.",
        "- This experiment does not establish "
        "physical causality.",
        "- Model checkpoints are selected by "
        "validation AP.",
        "- Hyperparameters were previously "
        "tuned on the same validation split.",
        "- Reported results are not independent "
        "test estimates.",
        "- Potential observational and processing "
        "artifacts require separate investigation.",
        "- The held-out test set remains unused.",
        "",
        "## 7. Reproduction",
        "",
        "Run from the repository root:",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\experiment_e1_views.py",
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

    print("Report:", REPORT)


def main():
    parser = argparse.ArgumentParser(
        description="E1: local/global TESS CNN ablation"
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Run first N of the nine experiments",
    )

    args = parser.parse_args()

    if args.max_trials is not None:
        if not 1 <= args.max_trials <= 9:
            parser.error(
                "--max-trials must be between 1 and 9"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    planned = [
        (mode, seed)
        for mode in MODES
        for seed in SEEDS
    ]

    subset = (
        planned[:args.max_trials]
        if args.max_trials
        else planned
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        f"E1 planned: {len(planned)} | "
        f"in this invocation: {len(subset)}"
    )

    trainer = None
    original_class = None

    for position, (mode, seed) in enumerate(
        subset,
        start=1,
    ):
        existing = get_result(
            mode,
            seed,
        )

        if existing is not None:
            print(
                f"[{position}/{len(subset)}] "
                f"SKIP {mode}, seed {seed}"
            )
            continue

        if trainer is None:
            trainer = grid.load_trainer()
            original_class = trainer.TessCNN

        cfg = configuration(
            mode,
            seed,
        )

        folder = trial_directory(
            mode,
            seed,
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_file = (
            folder / "trial_config.json"
        )

        if config_file.is_file():
            stored = json.loads(
                config_file.read_text(
                    encoding="utf-8"
                )
            )

            if stored != cfg:
                raise RuntimeError(
                    f"Configuration conflict: {folder}"
                )

        else:
            config_file.write_text(
                json.dumps(
                    cfg,
                    indent=2,
                ),
                encoding="utf-8",
            )

        trainer.TessCNN = make_model_class(
            original_class,
            mode,
        )

        print(
            f"\n[{position}/{len(subset)}] "
            f"TRAIN {mode}, seed {seed}",
            flush=True,
        )

        grid.run_trial(
            trainer,
            cfg,
            folder,
        )

        if get_result(mode, seed) is None:
            raise RuntimeError(
                f"Training produced no metrics: {folder}"
            )

    rows = [
        get_result(mode, seed)
        for mode, seed in planned
    ]

    rows = [
        row
        for row in rows
        if row is not None
    ]

    csv_path = write_results(rows)

    print(
        f"\nCompleted E1 experiments: "
        f"{len(rows)}/9"
    )

    print(
        "Results:",
        csv_path,
    )

    if len(rows) == 9:
        generate_report(rows)

        print("\nMEAN +/- SAMPLE SD")

        for metric in (
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "f1",
        ):
            print(metric)

            for mode in MODES:
                mean, sd = average_and_sd(
                    rows,
                    mode,
                    metric,
                )

                print(
                    f"  {mode:12s}: "
                    f"{mean:.4f} +/- {sd:.4f}"
                )

        print("E1 COMPLETE")

    else:
        print(
            "E1 incomplete: rerun "
            "without --max-trials"
        )

    print("TEST SET NOT USED")


if __name__ == "__main__":
    main()