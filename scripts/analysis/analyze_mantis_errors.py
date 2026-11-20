
"""
MODEL B - VALIDATION ERROR ANALYSIS

Compare original and tuned frozen-Mantis classifiers
across seeds 42, 123, and 2026.

Checks:
- Exact validation signal identifiers and labels
- TIC separation between train and validation
- Per-seed confusion matrices
- Persistent false positives and false negatives
- Signals improved or worsened by tuning

Does NOT:
- Train a model
- Change checkpoints
- Access the held-out test set

Run:
    python -u scripts/analysis/analyze_mantis_errors.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset" / "mantis_model_b"

OUTPUT = DATA / "error_analysis"

REPORT = ROOT / "docs" / "model_b_error_analysis.md"

SEEDS = (42, 123, 2026)

MODELS = {
    "Original": DATA / "mlp_head_v1",
    "Tuned": (
        DATA
        / "head_grid_v1"
        / "hidden_128__drop_0p5"
    ),
}

THRESHOLD = 0.5


def load_reference():
    path = DATA / "features_validation.npz"

    with np.load(path, allow_pickle=False) as data:
        uid = data["signal_uid"].astype(str)
        tic = data["tic_key"].astype(str)
        labels = data["y"].astype(int)

    if len(uid) != 293:
        raise ValueError("Unexpected validation size")

    if len(set(uid)) != len(uid):
        raise ValueError("Duplicate validation signal UID")

    if dict(
        zip(
            *np.unique(labels, return_counts=True)
        )
    ) != {0: 94, 1: 199}:
        raise ValueError("Unexpected validation labels")

    with np.load(
        DATA / "features_train.npz",
        allow_pickle=False,
    ) as train:
        train_tics = set(
            train["tic_key"].astype(str)
        )

    if train_tics.intersection(set(tic)):
        raise ValueError("Train-validation TIC leakage")

    return uid, tic, labels


def read_predictions(
    path,
    uid,
    tic,
    labels,
):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    required = {
        "signal_uid",
        "tic_key",
        "true_label",
        "predicted_probability",
    }

    if len(rows) != len(uid):
        raise ValueError(
            f"Incorrect prediction count: {path}"
        )

    if not rows or not required.issubset(
        rows[0].keys()
    ):
        raise ValueError(
            f"Missing CSV columns: {path}"
        )

    mapping = {}

    for row in rows:
        key = row["signal_uid"]

        if key in mapping:
            raise ValueError(
                f"Duplicate signal: {key}"
            )

        mapping[key] = row

    if set(mapping) != set(uid):
        raise ValueError(
            f"Prediction IDs do not match: {path}"
        )

    probabilities = []

    for identifier, star, label in zip(
        uid, tic, labels
    ):
        row = mapping[identifier]

        if row["tic_key"] != star:
            raise ValueError("TIC mismatch")

        if int(row["true_label"]) != int(label):
            raise ValueError("Label mismatch")

        probability = float(
            row["predicted_probability"]
        )

        if (
            not np.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "Invalid predicted probability"
            )

        probabilities.append(probability)

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


def evaluate(labels, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "ap": float(
            average_precision_score(
                labels, probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels, probabilities
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels, predictions
            )
        ),
        "f1": float(
            f1_score(labels, predictions)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("=" * 70)
    print("MODEL B - VALIDATION ERROR ANALYSIS")
    print("=" * 70)

    uid, tic, labels = load_reference()

    print("Validation signals:", len(uid))
    print("Confirmed planets:", int(sum(labels == 1)))
    print("False positives:", int(sum(labels == 0)))
    print("TIC leakage: NO")
    print("Held-out test set: NOT USED")

    output_rows = []
    probabilities_by_model = {}

    for model_name, directory in MODELS.items():
        runs = []

        for seed in SEEDS:
            path = (
                directory
                / f"seed_{seed}"
                / "predictions.csv"
            )

            runs.append(
                read_predictions(
                    path, uid, tic, labels
                )
            )

        probabilities_by_model[model_name] = (
            np.stack(runs)
        )

    print()
    print("PER-SEED RESULTS")
    print("-" * 70)

    report = [
        "# Model B - Validation Error Analysis",
        "",
        "## Dataset and protocol",
        "",
        "- 293 validation signals.",
        "- 199 confirmed planets (positive class).",
        "- 94 false positives (negative class).",
        "- Same TIC-separated validation data for both models.",
        "- Three random seeds per configuration.",
        "- Decision threshold fixed at 0.5.",
        "- Test set not used.",
        "",
        "## Per-seed confusion matrices",
        "",
        "| Model | Seed | AP | ROC-AUC | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name, runs in (
        probabilities_by_model.items()
    ):
        for seed, probabilities in zip(
            SEEDS, runs
        ):
            result = evaluate(
                labels, probabilities
            )

            print(
                f"{model_name:8s} seed={seed:4d} "
                f"AP={result['ap']:.4f} "
                f"FP={result['fp']:3d} "
                f"FN={result['fn']:3d}"
            )

            report.append(
                f"| {model_name} "
                f"| {seed} "
                f"| {result['ap']:.4f} "
                f"| {result['roc_auc']:.4f} "
                f"| {result['tn']} "
                f"| {result['fp']} "
                f"| {result['fn']} "
                f"| {result['tp']} |"
            )

    print()
    print("PERSISTENT ERRORS ACROSS ALL THREE SEEDS")
    print("-" * 70)

    report.extend([
        "",
        "## Persistent errors",
        "",
        "Persistent error = a signal misclassified",
        "in all three runs of the same configuration.",
        "",
        "| Model | Persistent FP | Persistent FN | Total |",
        "|---|---:|---:|---:|",
    ])

    wrong_by_model = {}

    for model_name, probabilities in (
        probabilities_by_model.items()
    ):
        predictions = (
            probabilities >= THRESHOLD
        )

        wrong = (
            predictions != labels[None, :]
        )

        wrong_count = wrong.sum(axis=0)

        wrong_by_model[model_name] = wrong_count

        persistent_fp = int(
            (
                (labels == 0)
                & (wrong_count == 3)
            ).sum()
        )

        persistent_fn = int(
            (
                (labels == 1)
                & (wrong_count == 3)
            ).sum()
        )

        print(
            f"{model_name}: "
            f"persistent FP={persistent_fp}, "
            f"persistent FN={persistent_fn}"
        )

        report.append(
            f"| {model_name} "
            f"| {persistent_fp} "
            f"| {persistent_fn} "
            f"| {persistent_fp + persistent_fn} |"
        )

    original_wrong = wrong_by_model["Original"]
    tuned_wrong = wrong_by_model["Tuned"]

    improved = int(
        (tuned_wrong < original_wrong).sum()
    )

    worsened = int(
        (tuned_wrong > original_wrong).sum()
    )

    unchanged = int(
        (tuned_wrong == original_wrong).sum()
    )

    print()
    print("COMPARISON OF ERROR FREQUENCY")
    print("-" * 70)
    print("Improved signals:", improved)
    print("Worsened signals:", worsened)
    print("Unchanged signals:", unchanged)

    report.extend([
        "",
        "## Changes after tuning",
        "",
        "For each validation signal, the number of",
        "incorrect predictions across three seeds",
        "is compared between the original and tuned head.",
        "",
        f"- Improved (fewer incorrect runs): {improved}",
        f"- Worsened (more incorrect runs): {worsened}",
        f"- Unchanged: {unchanged}",
        "",
        "## Methodological cautions",
        "",
        "- Errors are defined at a fixed 0.5 threshold.",
        "- Average Precision does not depend on this threshold.",
        "- Persistent errors may reflect ambiguous signals,",
        "  difficult transit shapes, missing coverage,",
        "  or incorrect labels; this audit cannot distinguish",
        "  those explanations without inspecting the light curves.",
        "- The validation set was used for model selection.",
        "- No test examples were inspected.",
        "",
    ])

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT / "validation_error_details.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow([
            "signal_uid",
            "tic_key",
            "true_label",
            "original_mean_probability",
            "tuned_mean_probability",
            "original_errors_out_of_3",
            "tuned_errors_out_of_3",
        ])

        original_mean = (
            probabilities_by_model["Original"]
            .mean(axis=0)
        )

        tuned_mean = (
            probabilities_by_model["Tuned"]
            .mean(axis=0)
        )

        for i in range(len(uid)):
            writer.writerow([
                uid[i],
                tic[i],
                int(labels[i]),
                float(original_mean[i]),
                float(tuned_mean[i]),
                int(original_wrong[i]),
                int(tuned_wrong[i]),
            ])

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("CSV:", csv_path)
    print("Report:", REPORT)
    print("TEST SET NOT USED")
    print("ERROR ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
