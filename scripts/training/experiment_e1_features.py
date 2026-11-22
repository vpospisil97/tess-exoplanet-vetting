
"""
TESS Experiment E1 - Feature-level view ablation.

Improved control experiment for local vs global views.

The disabled view is zeroed AFTER its CNN encoder,
not at the original input.

The original E1 and all earlier results are preserved.

Run:
    python -u scripts/training/experiment_e1_features.py --max-trials 1
    python -u scripts/training/experiment_e1_features.py
"""

from pathlib import Path
import statistics

import torch

import experiment_e1_views as e1


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT / "full_dataset"
    / "cnn_e1_feature_ablation"
)

REPORT = (
    ROOT / "docs"
    / "model_a_experiment_e1_features.md"
)

SEEDS = (42, 123, 2026)

MODES = (
    "local_only",
    "global_only",
    "both",
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


def feature_ablation_class(original_class, mode):
    """
    Keep the same declared network architecture.

    Only encoder features from the selected
    view(s) can reach the classification head.

    Inactive encoder features are replaced
    with zeros during training and validation.
    """

    if mode not in MODES:
        raise ValueError(
            f"Unknown view mode: {mode}"
        )

    class FeatureAblationCNN(original_class):

        def forward(self, local, global_view):

            """Vypočítá predikci se zapojením zvolených příznaků."""
            local_features = self.local_encoder(
                local
            )

            global_features = self.global_encoder(
                global_view
            )

            if mode == "local_only":
                global_features = torch.zeros_like(
                    global_features
                )

            elif mode == "global_only":
                local_features = torch.zeros_like(
                    local_features
                )

            features = torch.cat(
                [
                    local_features,
                    global_features,
                ],
                dim=1,
            )

            logits = self.classifier(
                features
            )

            return logits.squeeze(1)

    return FeatureAblationCNN


def create_report(rows):
    """Uloží validační výsledky ablace příznaků E1."""
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Experiment E1 - Feature-level ablation",
        "",
        "## Research question",
        "",
        "How much does the TESS classifier benefit "
        "from combining local and global "
        "phase-folded light-curve views?",
        "",
        "## Hypothesis",
        "",
        "Using both views provides higher validation "
        "Average Precision than using either "
        "view alone.",
        "",
        "## Experimental design",
        "",
        "- Same dataset and TIC-based split.",
        "- Same input preprocessing.",
        "- Same two-branch CNN definition.",
        "- Same training hyperparameters.",
        "- Three seeds: 42, 123, 2026.",
        "- Learning rate: 0.001.",
        "- Batch size: 32.",
        "- Dropout: 0.3.",
        "- Weight decay: 0.0001.",
        "- Maximum epochs: 35.",
        "- Early stopping patience: 7.",
        "- Checkpoint selection: validation AP.",
        "- Test set not evaluated.",
        "",
        "## Ablation method",
        "",
        "Each input view is encoded by its "
        "own convolutional encoder.",
        "",
        "- Both: retain both encoder outputs.",
        "- Local only: replace global encoder "
        "features with zeros.",
        "- Global only: replace local encoder "
        "features with zeros.",
        "",
        "The feature masking is applied in both "
        "training and validation.",
        "",
        "This prevents inactive encoders from "
        "contributing constant features generated "
        "by convolution biases or BatchNorm.",
        "",
        "Although the declared model has the "
        "same number of parameters, inactive "
        "branches receive no supervised "
        "gradient and are not functionally used.",
        "",
        "## Individual runs",
        "",
        "| Mode | Seed | Best epoch | AP | "
        "ROC-AUC | Balanced accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['mode']} "
            f"| {row['seed']} "
            f"| {row['best_epoch']} "
            f"| {row['average_precision']:.4f} "
            f"| {row['roc_auc']:.4f} "
            f"| {row['balanced_accuracy']:.4f} "
            f"| {row['f1']:.4f} |"
        )

    lines.extend([
        "",
        "## Mean and sample standard deviation",
        "",
        "| Metric | Both | Local only | Global only |",
        "|---|---:|---:|---:|",
    ])

    report_order = (
        "both",
        "local_only",
        "global_only",
    )

    for metric in METRICS:
        cells = []

        for mode in report_order:
            values = [
                row[metric]
                for row in rows
                if row["mode"] == mode
            ]

            if len(values) != 3:
                raise RuntimeError(
                    f"Missing runs: {mode}, {metric}"
                )

            mean = statistics.mean(values)
            std = statistics.stdev(values)

            cells.append(
                f"{mean:.4f} +/- {std:.4f}"
            )

        lines.append(
            f"| {metric} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend([
        "",
        "## Paired Average Precision differences",
        "",
        "| Seed | Both - Local | Both - Global |",
        "|---:|---:|---:|",
    ])

    for seed in SEEDS:
        lookup = {
            row["mode"]: row
            for row in rows
            if row["seed"] == seed
        }

        both = lookup["both"]["average_precision"]

        local = lookup["local_only"][
            "average_precision"
        ]

        global_ap = lookup["global_only"][
            "average_precision"
        ]

        lines.append(
            f"| {seed} "
            f"| {both - local:+.4f} "
            f"| {both - global_ap:+.4f} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is a feature-ablation study.",
        "- Disabled branches do not contribute "
        "features or useful trainable capacity.",
        "- The compared variants therefore have "
        "different effective model capacities.",
        "- Fixed-threshold F1 can be affected "
        "by probability calibration.",
        "- AP and ROC-AUC measure ranking "
        "without choosing a threshold.",
        "- Hyperparameters were already selected "
        "on the same validation split.",
        "- Results are not independent "
        "test-set estimates.",
        "- Processing artifacts and dataset "
        "biases remain possible.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\experiment_e1_features.py",
        "```",
        "",
    ])

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nReport created:", REPORT)


def main():
    """Spustí ablační experiment lokálních a globálních příznaků."""
    original_configuration = e1.configuration

    def tagged_configuration(mode, seed):
        """Označí konfiguraci experimentu podle varianty a seedu."""
        config = original_configuration(
            mode,
            seed,
        )

        config["ablation_method"] = (
            "encoder_feature_zeroing"
        )

        config["experiment_version"] = 2

        return config

    # Separate outputs preserve the original E1.
    e1.OUTPUT = OUTPUT
    e1.REPORT = REPORT

    # Run the problematic single-view condition first.
    e1.MODES = MODES

    # Replace only the view-ablation implementation.
    e1.make_model_class = feature_ablation_class

    # Record the exact experimental variant.
    e1.configuration = tagged_configuration

    # Generate an accurate report for this variant.
    e1.generate_report = create_report

    # Reuse the previously tested training,
    # checkpoint and results infrastructure.
    e1.main()


if __name__ == "__main__":
    main()
