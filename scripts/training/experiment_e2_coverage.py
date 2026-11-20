
"""
TESS Deep Learning - Experiment E2
==================================

Research question:
Does the explicit coverage mask improve classification
of TESS exoplanet transit signals?

Two experimental conditions:

1. with_mask
   The CNN receives both channels:
   - transformed flux
   - coverage mask

2. without_mask
   The CNN receives the same flux values,
   but the explicit coverage channel is zeroed.

Both conditions:
- use the same CNN architecture
- use the same TIC-based train/validation split
- use identical hyperparameters
- are repeated with seeds 42, 123 and 2026
- never evaluate the held-out test set

Run from the project root:

    python -u scripts/training/experiment_e2_coverage.py --max-trials 1

Then:

    python -u scripts/training/experiment_e2_coverage.py

Results:
    full_dataset/cnn_e2_coverage/

Report:
    docs/model_a_experiment_e2_coverage.md
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

import grid_search_cnn as grid


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "full_dataset"
    / "cnn_e2_coverage"
)

REPORT = (
    ROOT
    / "docs"
    / "model_a_experiment_e2_coverage.md"
)

MODES = (
    "with_mask",
    "without_mask",
)

SEEDS = (
    42,
    123,
    2026,
)

METRICS = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "precision",
    "recall",
)

FIELDS = (
    "mode",
    "seed",
    "best_epoch",
    *METRICS,
    "training_seconds",
    "trainable_parameters",
)


# ============================================================
# CONFIGURATION
# ============================================================

def config_for(mode, seed):

    if mode not in MODES:
        raise ValueError(
            f"Unknown experiment mode: {mode}"
        )

    if seed not in SEEDS:
        raise ValueError(
            f"Unknown random seed: {seed}"
        )

    return {
        "id": f"e2_{mode}_seed_{seed}",
        "mode": mode,
        "seed": seed,
        "epochs": 35,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.3,
        "patience": 7,
        "device": "cuda",
        "ablation_method": "coverage_channel_zeroing",
    }


def trial_directory(mode, seed):

    return (
        OUTPUT
        / mode
        / f"seed_{seed}"
    )


# ============================================================
# MODEL ABLATION
# ============================================================

def make_model_class(original_class, mode):
    """
    Create a CNN variant with the same architecture.

    Input tensors have shape:

        local:  [batch, 2, 101]
        global: [batch, 2, 201]

    Channel 0: transformed flux
    Channel 1: explicit coverage mask

    In the without_mask condition, only channel 1
    is replaced with zeros.

    Flux values remain unchanged.
    """

    if mode not in MODES:
        raise ValueError(
            f"Invalid mode: {mode}"
        )

    class CoverageAblationCNN(original_class):

        def forward(self, local, global_view):

            if mode == "without_mask":

                local_flux = local[:, :1, :]

                local_mask = torch.zeros_like(
                    local[:, 1:2, :]
                )

                global_flux = global_view[:, :1, :]

                global_mask = torch.zeros_like(
                    global_view[:, 1:2, :]
                )

                local = torch.cat(
                    (
                        local_flux,
                        local_mask,
                    ),
                    dim=1,
                )

                global_view = torch.cat(
                    (
                        global_flux,
                        global_mask,
                    ),
                    dim=1,
                )

            return super().forward(
                local,
                global_view,
            )

    return CoverageAblationCNN


# ============================================================
# READ EXISTING RESULTS
# ============================================================

def read_run(mode, seed):

    directory = trial_directory(
        mode,
        seed,
    )

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

        if checkpoint_path.exists():
            raise RuntimeError(
                "Checkpoint exists without metrics: "
                f"{directory}"
            )

        return None

    if not config_path.is_file():
        raise RuntimeError(
            f"Missing configuration: {directory}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Missing checkpoint: {directory}"
        )

    expected = config_for(
        mode,
        seed,
    )

    stored = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if stored != expected:
        raise RuntimeError(
            "Stored experiment configuration "
            f"does not match: {directory}"
        )

    result = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    if result.get("test_set_used") is not False:
        raise RuntimeError(
            f"Test data used unexpectedly: {directory}"
        )

    if result.get("seed") != seed:
        raise RuntimeError(
            f"Incorrect seed: {directory}"
        )

    if result.get("model") != "two_view_1d_cnn":
        raise RuntimeError(
            f"Unexpected model: {directory}"
        )

    expected_parameters = 79937

    if (
        result.get("trainable_parameters")
        != expected_parameters
    ):
        raise RuntimeError(
            f"Incorrect parameter count: {directory}"
        )

    hyperparameters = result["hyperparameters"]

    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "patience",
        "device",
    ):

        if hyperparameters.get(key) != expected[key]:
            raise RuntimeError(
                "Hyperparameter mismatch "
                f"({key}): {directory}"
            )

    validation = result["validation"]

    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(
            result["best_epoch"]
        ),
        **{
            metric: float(validation[metric])
            for metric in METRICS
        },
        "training_seconds": float(
            result["training_seconds"]
        ),
        "trainable_parameters": int(
            result["trainable_parameters"]
        ),
    }


# ============================================================
# STATISTICS
# ============================================================

def mean_and_sd(rows, mode, metric):

    values = [
        row[metric]
        for row in rows
        if row["mode"] == mode
    ]

    if len(values) != len(SEEDS):
        raise RuntimeError(
            "Missing results for "
            f"{mode}, metric {metric}"
        )

    mean = statistics.mean(values)

    sd = statistics.stdev(values)

    return mean, sd


# ============================================================
# SAVE CSV
# ============================================================

def save_results(rows):

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT / "e2_results.csv"
    )

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


# ============================================================
# GENERATE MARKDOWN REPORT
# ============================================================

def generate_report(rows):

    lines = [
        "# Experiment E2 - Coverage Mask Ablation",
        "",
        "## 1. Research question",
        "",
        "Does the explicit coverage mask improve "
        "the classification of TESS transit signals?",
        "",
        "## 2. Hypothesis",
        "",
        "Providing an explicit binary coverage "
        "channel improves validation Average "
        "Precision compared with using "
        "the flux channels alone.",
        "",
        "## 3. Experimental setup",
        "",
        "- Dataset: TESS transit signals.",
        "- Same TIC-based data split.",
        "- Same CNN architecture.",
        "- Trainable parameters: 79,937.",
        "- Same flux preprocessing.",
        "- Learning rate: 0.001.",
        "- Dropout: 0.3.",
        "- Batch size: 32.",
        "- Weight decay: 0.0001.",
        "- Maximum epochs: 35.",
        "- Early stopping patience: 7.",
        "- Random seeds: 42, 123, 2026.",
        "- Checkpoint selection: validation AP.",
        "- Test set was not evaluated.",
        "",
        "## 4. Experimental conditions",
        "",
        "### With mask",
        "",
        "The model receives both input channels:",
        "",
        "- Transformed flux.",
        "- Explicit binary coverage mask.",
        "",
        "### Without mask",
        "",
        "The flux channel remains unchanged.",
        "",
        "The explicit coverage channel is replaced "
        "with zeros before convolutional encoding.",
        "",
        "The same masking is applied during "
        "training and validation.",
        "",
        "## 5. Individual results",
        "",
        "| Mode | Seed | Best epoch | AP | "
        "ROC-AUC | Balanced accuracy | F1 | Time (s) |",
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
        "## 6. Mean and sample standard deviation",
        "",
        "| Metric | With mask | Without mask | Difference |",
        "|---|---:|---:|---:|",
    ])

    for metric in METRICS:

        with_mean, with_sd = mean_and_sd(
            rows,
            "with_mask",
            metric,
        )

        without_mean, without_sd = mean_and_sd(
            rows,
            "without_mask",
            metric,
        )

        difference = with_mean - without_mean

        lines.append(
            f"| {metric} "
            f"| {with_mean:.4f} +/- {with_sd:.4f} "
            f"| {without_mean:.4f} +/- {without_sd:.4f} "
            f"| {difference:+.4f} |"
        )

    lines.extend([
        "",
        "## 7. Paired AP differences",
        "",
        "| Seed | With mask | Without mask | Difference |",
        "|---:|---:|---:|---:|",
    ])

    for seed in SEEDS:

        with_run = next(
            row
            for row in rows
            if row["mode"] == "with_mask"
            and row["seed"] == seed
        )

        without_run = next(
            row
            for row in rows
            if row["mode"] == "without_mask"
            and row["seed"] == seed
        )

        with_ap = with_run["average_precision"]

        without_ap = without_run["average_precision"]

        lines.append(
            f"| {seed} "
            f"| {with_ap:.4f} "
            f"| {without_ap:.4f} "
            f"| {with_ap - without_ap:+.4f} |"
        )

    lines.extend([
        "",
        "## 8. Interpretation and limitations",
        "",
        "- Average Precision is the main metric.",
        "- Both variants use identical "
        "data splits and training settings.",
        "- Both variants have the same declared "
        "network architecture.",
        "- The missing explicit coverage channel "
        "reduces usable input information.",
        "- Missing bins may still be identifiable "
        "from zero-filled flux values.",
        "- Therefore this is not complete removal "
        "of all missing-data information.",
        "- Mask information may encode useful "
        "coverage or unwanted observational bias.",
        "- The model was previously tuned on "
        "the same validation split.",
        "- Three seeds measure sensitivity "
        "to initialization, not independent "
        "generalization performance.",
        "- The held-out test data remains unused.",
        "",
        "## 9. Reproduction",
        "",
        "Run from repository root:",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\experiment_e2_coverage.py",
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

    print(
        "Report created:",
        REPORT,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Experiment E2: coverage mask "
            "ablation for the TESS CNN"
        )
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Run the first N experiments out of six",
    )

    args = parser.parse_args()

    if args.max_trials is not None:

        if not 1 <= args.max_trials <= 6:
            parser.error(
                "--max-trials must be between 1 and 6"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    print("=" * 72)
    print("TESS - EXPERIMENT E2 - COVERAGE MASK")
    print("=" * 72)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    planned = [
        (mode, seed)
        for mode in MODES
        for seed in SEEDS
    ]

    selected = (
        planned
        if args.max_trials is None
        else planned[:args.max_trials]
    )

    print(
        f"Total experiments: {len(planned)}"
    )

    print(
        f"Experiments in this run: {len(selected)}"
    )

    trainer = None
    original_class = None

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for index, (mode, seed) in enumerate(
        selected,
        start=1,
    ):

        existing = read_run(
            mode,
            seed,
        )

        if existing is not None:

            print(
                f"[{index}/{len(selected)}] "
                f"SKIP {mode}, seed {seed}"
            )

            continue

        if trainer is None:

            trainer = grid.load_trainer()

            original_class = trainer.TessCNN

        config = config_for(
            mode,
            seed,
        )

        directory = trial_directory(
            mode,
            seed,
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_path = (
            directory / "trial_config.json"
        )

        if config_path.exists():

            stored = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

            if stored != config:
                raise RuntimeError(
                    "Configuration conflict: "
                    f"{directory}"
                )

        else:

            config_path.write_text(
                json.dumps(
                    config,
                    indent=2,
                ),
                encoding="utf-8",
            )

        trainer.TessCNN = make_model_class(
            original_class,
            mode,
        )

        print()
        print("=" * 72)

        print(
            f"[{index}/{len(selected)}] "
            f"TRAIN {mode}, seed {seed}"
        )

        print("=" * 72)

        grid.run_trial(
            trainer,
            config,
            directory,
        )

        if read_run(mode, seed) is None:
            raise RuntimeError(
                "Training did not produce "
                f"complete results: {directory}"
            )

    # --------------------------------------------------------
    # COLLECT RESULTS
    # --------------------------------------------------------

    rows = [
        read_run(mode, seed)
        for mode, seed in planned
    ]

    rows = [
        row
        for row in rows
        if row is not None
    ]

    csv_path = save_results(rows)

    print()
    print("=" * 72)
    print("E2 RESULTS")
    print("=" * 72)

    print(
        f"Completed experiments: {len(rows)}/6"
    )

    print(
        "CSV:",
        csv_path,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    if len(rows) == len(planned):

        generate_report(rows)

        print()
        print("MEAN +/- SAMPLE SD")
        print("-" * 72)

        for metric in (
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "f1",
        ):

            print(metric)

            for mode in MODES:

                mean, sd = mean_and_sd(
                    rows,
                    mode,
                    metric,
                )

                print(
                    f"  {mode:14s}: "
                    f"{mean:.4f} +/- {sd:.4f}"
                )

        print()
        print("E2 COMPLETE")

    else:

        print()
        print(
            "E2 incomplete. "
            "Run again without --max-trials."
        )

    print("TEST SET NOT USED")
    print("DONE")


if __name__ == "__main__":
    main()
