"""Sestavuje validační zprávu o CNN a jejím srovnání s baseline."""


from pathlib import Path
import csv
import json
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset"
CNN = DATA / "cnn_model_a"
DOCS = ROOT / "docs"

OUTPUT = DOCS / "model_a_validation.md"

SEEDS = (42, 123, 2026)

METRICS = {
    "average_precision": "Average Precision",
    "roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1-score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
}


def load_json(path):
    """Bezpečně načte uložený JSON soubor s výsledky."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_baseline():
    """Načte validační metriky logistické regrese."""
    path = DATA / "baseline_validation_metrics.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(csv.DictReader(file))

    matches = [
        row
        for row in rows
        if row.get("model") == "logistic_regression"
        and row.get("split") == "validation"
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected one logistic regression validation row"
        )

    return matches[0]


def load_cnn_results():
    """Načte a ověří výsledky opakovaných běhů CNN."""
    results = []

    for seed in SEEDS:
        path = CNN / f"metrics_seed_{seed}.json"

        result = load_json(path)

        if int(result["seed"]) != seed:
            raise ValueError(
                f"Unexpected seed in {path}"
            )

        if result["test_set_used"] is not False:
            raise ValueError(
                f"Test set was used in seed {seed}"
            )

        if result["model"] != "two_view_1d_cnn":
            raise ValueError(
                f"Unexpected model in seed {seed}"
            )

        results.append(result)

    reference = results[0]

    for result in results[1:]:
        if (
            result["train_signals"]
            != reference["train_signals"]
        ):
            raise ValueError(
                "Training sample counts differ"
            )

        if (
            result["validation_signals"]
            != reference["validation_signals"]
        ):
            raise ValueError(
                "Validation sample counts differ"
            )

        if (
            result["trainable_parameters"]
            != reference["trainable_parameters"]
        ):
            raise ValueError(
                "Model parameter counts differ"
            )

        reference_config = {
            key: value
            for key, value
            in reference["hyperparameters"].items()
            if key != "seed"
        }

        current_config = {
            key: value
            for key, value
            in result["hyperparameters"].items()
            if key != "seed"
        }

        if current_config != reference_config:
            raise ValueError(
                "Hyperparameters differ between seeds"
            )

    return results


def verify_dataset():
    """Ověří počty vzorků a vazbu výsledků na připravená data."""
    views_path = DATA / "full_cnn_views.npz"
    split_path = DATA / "cnn_split_indices.npz"

    if not views_path.is_file():
        raise FileNotFoundError(views_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    with np.load(
        views_path,
        allow_pickle=False,
    ) as views:

        labels = views["y"].astype(int)
        tics = views["tic_key"].astype(str)

    with np.load(
        split_path,
        allow_pickle=False,
    ) as splits:

        indices = {
            "train": splits["train_idx"].astype(int),
            "validation": splits["validation_idx"].astype(int),
            "test": splits["test_idx"].astype(int),
        }

    all_indices = np.concatenate(
        list(indices.values())
    )

    if not np.array_equal(
        np.sort(all_indices),
        np.arange(len(labels)),
    ):
        raise ValueError(
            "Dataset splits overlap or omit samples"
        )

    tic_sets = {
        name: set(tics[idx])
        for name, idx in indices.items()
    }

    if (
        tic_sets["train"] & tic_sets["validation"]
        or tic_sets["train"] & tic_sets["test"]
        or tic_sets["validation"] & tic_sets["test"]
    ):
        raise ValueError(
            "TIC leakage detected"
        )

    summary = {}

    for name, idx in indices.items():
        subset = labels[idx]

        summary[name] = {
            "signals": len(idx),
            "tics": len(tic_sets[name]),
            "confirmed": int(
                np.sum(subset == 1)
            ),
            "false_positive": int(
                np.sum(subset == 0)
            ),
        }

    return summary


def mean_std(values):
    """Vypočítá průměr a výběrovou směrodatnou odchylku."""
    values = [float(v) for v in values]

    return (
        statistics.mean(values),
        statistics.stdev(values),
    )


def format_number(value):
    """Převede numerickou metriku na jednotný textový formát."""
    return f"{float(value):.4f}"


def main():
    """Vytvoří souhrnnou validační zprávu modelu A."""
    print("=" * 70)
    print("TESS - MODEL A VALIDATION REPORT")
    print("=" * 70)

    baseline = load_baseline()
    results = load_cnn_results()
    dataset = verify_dataset()

    DOCS.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    lines.append("# Model A - Two-view 1D CNN")
    lines.append("")
    lines.append(
        "## Validation results - three random seeds"
    )
    lines.append("")

    lines.append(
        "This report was generated automatically "
        "from local experiment results."
    )

    lines.append("")
    lines.append(
        "**Task:** binary classification of TESS "
        "transit signals."
    )
    lines.append("")
    lines.append(
        "- Positive class (1): confirmed"
    )
    lines.append(
        "- Negative class (0): false_positive"
    )
    lines.append(
        "- Model selection metric: validation Average Precision"
    )
    lines.append(
        "- Test set: not used for model selection or evaluation"
    )

    lines.append("")
    lines.append("## 1. Dataset")
    lines.append("")

    lines.append(
        "| Split | Signals | Unique TIC | "
        "Confirmed | False positive |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for name in ("train", "validation", "test"):
        row = dataset[name]

        lines.append(
            f"| {name} "
            f"| {row['signals']} "
            f"| {row['tics']} "
            f"| {row['confirmed']} "
            f"| {row['false_positive']} |"
        )

    lines.append("")
    lines.append(
        "Splits were verified to have no overlap "
        "at the TIC (star) level."
    )

    lines.append("")
    lines.append("## 2. Model architecture")
    lines.append("")

    lines.append(
        "The model consists of two separate "
        "one-dimensional convolutional encoders."
    )

    lines.append("")
    lines.append(
        "- Local view: 101 time bins, flux and coverage mask"
    )
    lines.append(
        "- Global view: 201 time bins, flux and coverage mask"
    )
    lines.append(
        "- Convolutional layers: 32, 64, and 64 channels per encoder"
    )
    lines.append(
        "- Pooling: adaptive average and maximum pooling"
    )
    lines.append(
        "- Classification head: 256 -> 128 -> 1"
    )

    lines.append(
        f"- Trainable parameters: "
        f"{results[0]['trainable_parameters']:,}"
    )

    lines.append("")
    lines.append("The model is trained from random initialization.")
    lines.append("")
    lines.append(
        "Input transformation: "
        "`arcsinh(flux_ppt / 5)` with coverage masking."
    )

    lines.append("")
    lines.append("## 3. Hyperparameters")
    lines.append("")

    config = results[0]["hyperparameters"]

    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "patience",
        "device",
    ):
        if key in config:
            lines.append(
                f"- `{key}`: `{config[key]}`"
            )

    lines.append("")
    lines.append(
        "Optimizer: AdamW. Loss: weighted BCEWithLogitsLoss."
    )

    lines.append("")
    lines.append(
        "Positive-class weight is calculated "
        "from the training labels."
    )

    lines.append("")
    lines.append("## 4. Individual runs")
    lines.append("")

    lines.append(
        "| Seed | Best epoch | Epochs run | AP | "
        "ROC-AUC | Balanced Accuracy | F1 | Time (s) |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for result in results:
        m = result["validation"]

        lines.append(
            f"| {result['seed']} "
            f"| {result['best_epoch']} "
            f"| {result['epochs_completed']} "
            f"| {format_number(m['average_precision'])} "
            f"| {format_number(m['roc_auc'])} "
            f"| {format_number(m['balanced_accuracy'])} "
            f"| {format_number(m['f1'])} "
            f"| {result['training_seconds']:.2f} |"
        )

    lines.append("")
    lines.append(
        "Each checkpoint was selected by the "
        "highest validation Average Precision."
    )

    lines.append("")
    lines.append("## 5. Mean and standard deviation")
    lines.append("")

    lines.append(
        "| Metric | Logistic regression baseline | "
        "CNN mean | CNN sample SD |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    for key, label in METRICS.items():
        values = [
            result["validation"][key]
            for result in results
        ]

        mean, std = mean_std(values)

        lines.append(
            f"| {label} "
            f"| {format_number(baseline[key])} "
            f"| {format_number(mean)} "
            f"| {format_number(std)} |"
        )

    lines.append("")
    lines.append(
        "The standard deviation describes variation "
        "across three CNN random seeds on the same "
        "validation split. The baseline was evaluated "
        "in one run."
    )

    lines.append("")
    lines.append("## 6. Confusion matrices")
    lines.append("")

    lines.append(
        "| Seed | TN | FP | FN | TP |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|"
    )

    for result in results:
        m = result["validation"]

        lines.append(
            f"| {result['seed']} "
            f"| {m['tn']} "
            f"| {m['fp']} "
            f"| {m['fn']} "
            f"| {m['tp']} |"
        )

    lines.append("")
    lines.append(
        "Confusion matrices use a fixed "
        "classification threshold of 0.5."
    )

    lines.append("")
    lines.append("## 7. Reproduction")
    lines.append("")
    lines.append(
        "Run from the repository root, with the "
        "prepared dataset available locally."
    )

    lines.append("")
    lines.append("```powershell")

    for seed in SEEDS:
        lines.append(
            "python -u .\\run.py train_cnn "
            "--epochs 35 --batch-size 64 "
            "--lr 0.001 --patience 7 "
            f"--seed {seed} --device cuda"
        )

    lines.append("```")
    lines.append("")

    lines.append(
        "Generate this report using:"
    )
    lines.append("")
    lines.append("```powershell")
    lines.append(
        "python .\\scripts\\analysis\\generate_model_a_report.py"
    )
    lines.append("```")

    lines.append("")
    lines.append("## 8. Limitations and next steps")
    lines.append("")

    lines.append(
        "- Results are validation results, not "
        "independent test-set estimates."
    )

    lines.append(
        "- The checkpoint is selected using the "
        "same validation set reported here."
    )

    lines.append(
        "- A separate hyperparameter search "
        "and final evaluation are still required."
    )

    lines.append(
        "- The source of predictive information "
        "must be checked for metadata, processing, "
        "or observational artifacts."
    )

    lines.append(
        "- The pretrained/adapted Model B "
        "has not yet been evaluated."
    )

    lines.append(
        "- The current results do not establish "
        "statistical significance versus the baseline."
    )

    OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nDataset:")
    for name, row in dataset.items():
        print(name, row)

    print("\nCNN summary:")

    for key in (
        "average_precision",
        "roc_auc",
        "balanced_accuracy",
        "f1",
    ):
        values = [
            result["validation"][key]
            for result in results
        ]

        mean, std = mean_std(values)

        print(
            f"{key:20s} "
            f"{mean:.4f} +/- {std:.4f}"
        )

    print("\nReport:", OUTPUT)
    print("TEST SET NOT EVALUATED")
    print("DONE")


if __name__ == "__main__":
    main()
