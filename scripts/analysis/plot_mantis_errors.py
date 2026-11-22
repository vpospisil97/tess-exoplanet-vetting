
"""
MODEL B - PLOT PERSISTENT VALIDATION ERRORS

Creates figures showing actual TESS light-curve
views for persistent errors of the tuned Mantis model.

Selection:
  - 4 persistent false positives, highest mean probability
  - 4 persistent false negatives, lowest mean probability

Both local and global views are plotted.
Coverage masks are displayed separately on a second axis.

Uses only previously identified validation examples.
Does not train models or evaluate the test set.

Run:
  python -u scripts/analysis/plot_mantis_errors.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT / "full_dataset" / "full_cnn_views.npz"
)

FEATURES = (
    ROOT
    / "full_dataset"
    / "mantis_model_b"
    / "features_validation.npz"
)

ERRORS = (
    ROOT
    / "full_dataset"
    / "mantis_model_b"
    / "error_analysis"
    / "validation_error_details.csv"
)

FIGURES = ROOT / "docs" / "figures"

REPORT = (
    ROOT
    / "docs"
    / "model_b_error_examples.md"
)

NUMBER_PER_GROUP = 4


def load_validation():
    """Načte validační reprezentace a odpovídající světelné křivky."""
    print("Loading validation references...")

    with np.load(
        FEATURES,
        allow_pickle=False,
    ) as data:
        indices = np.asarray(
            data["indices"],
            dtype=np.int64,
        )

        uid = data["signal_uid"].astype(str)
        tic = data["tic_key"].astype(str)
        labels = data["y"].astype(np.int64)

    if len(indices) != 293:
        raise ValueError(
            "Expected 293 validation signals"
        )

    if len(set(indices.tolist())) != len(indices):
        raise ValueError(
            "Duplicate validation indices"
        )

    with np.load(
        DATASET,
        allow_pickle=False,
    ) as data:
        required = (
            "x_local",
            "x_global",
            "signal_uid",
            "tic_key",
            "y",
        )

        missing = [
            key
            for key in required
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Missing dataset fields: {missing}"
            )

        if data["x_local"].shape != (1918, 101, 2):
            raise ValueError(
                "Unexpected local view shape"
            )

        if data["x_global"].shape != (1918, 201, 2):
            raise ValueError(
                "Unexpected global view shape"
            )

        raw_uid = data["signal_uid"][indices].astype(str)
        raw_tic = data["tic_key"][indices].astype(str)
        raw_labels = data["y"][indices].astype(np.int64)

        if not np.array_equal(uid, raw_uid):
            raise ValueError(
                "Signal UID alignment mismatch"
            )

        if not np.array_equal(tic, raw_tic):
            raise ValueError(
                "TIC alignment mismatch"
            )

        if not np.array_equal(labels, raw_labels):
            raise ValueError(
                "Label alignment mismatch"
            )

        local = np.asarray(
            data["x_local"][indices],
            dtype=np.float32,
        )

        global_view = np.asarray(
            data["x_global"][indices],
            dtype=np.float32,
        )

    if (
        not np.isfinite(local).all()
        or not np.isfinite(global_view).all()
    ):
        raise ValueError(
            "Non-finite input views"
        )

    return {
        "uid": uid,
        "tic": tic,
        "labels": labels,
        "local": local,
        "global": global_view,
    }


def load_error_details(validation):
    """Načte informace o opakovaných chybách modelu Mantis."""
    print("Loading error analysis...")

    if not ERRORS.is_file():
        raise FileNotFoundError(ERRORS)

    with ERRORS.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 293:
        raise ValueError(
            "Expected 293 error-detail records"
        )

    mapping = {}

    for row in rows:
        identifier = row["signal_uid"]

        if identifier in mapping:
            raise ValueError(
                f"Duplicate signal UID: {identifier}"
            )

        mapping[identifier] = row

    if set(mapping) != set(validation["uid"]):
        raise ValueError(
            "Error-detail signals do not match validation"
        )

    examples = []

    for position, identifier in enumerate(
        validation["uid"]
    ):
        row = mapping[identifier]

        tic = validation["tic"][position]
        label = int(
            validation["labels"][position]
        )

        if row["tic_key"] != tic:
            raise ValueError(
                f"TIC mismatch for {identifier}"
            )

        if int(row["true_label"]) != label:
            raise ValueError(
                f"Label mismatch for {identifier}"
            )

        original_probability = float(
            row["original_mean_probability"]
        )

        tuned_probability = float(
            row["tuned_mean_probability"]
        )

        original_errors = int(
            row["original_errors_out_of_3"]
        )

        tuned_errors = int(
            row["tuned_errors_out_of_3"]
        )

        if not (
            0 <= original_probability <= 1
            and 0 <= tuned_probability <= 1
        ):
            raise ValueError(
                "Invalid probabilities"
            )

        if not (
            0 <= original_errors <= 3
            and 0 <= tuned_errors <= 3
        ):
            raise ValueError(
                "Invalid error counts"
            )

        examples.append({
            "position": position,
            "signal_uid": identifier,
            "tic_key": tic,
            "label": label,
            "original_probability": original_probability,
            "tuned_probability": tuned_probability,
            "original_errors": original_errors,
            "tuned_errors": tuned_errors,
        })

    return examples


def select_examples(examples):
    """Vybere ukázky opakovaných falešně pozitivních a negativních klasifikací."""
    false_positives = [
        example
        for example in examples
        if (
            example["label"] == 0
            and example["tuned_errors"] == 3
        )
    ]

    false_negatives = [
        example
        for example in examples
        if (
            example["label"] == 1
            and example["tuned_errors"] == 3
        )
    ]

    false_positives.sort(
        key=lambda example: (
            -example["tuned_probability"],
            example["signal_uid"],
        )
    )

    false_negatives.sort(
        key=lambda example: (
            example["tuned_probability"],
            example["signal_uid"],
        )
    )

    print(
        "Persistent false positives:",
        len(false_positives),
    )

    print(
        "Persistent false negatives:",
        len(false_negatives),
    )

    if (
        len(false_positives) < NUMBER_PER_GROUP
        or len(false_negatives) < NUMBER_PER_GROUP
    ):
        raise RuntimeError(
            "Not enough persistent errors for plotting"
        )

    return (
        false_positives[:NUMBER_PER_GROUP],
        false_negatives[:NUMBER_PER_GROUP],
    )


def plot_view(ax, view, title):
    """Vykreslí světelný tok a pokrytí jednoho fázového pohledu."""
    flux = view[:, 0]
    coverage = view[:, 1]

    x = np.arange(len(flux))

    ax.plot(
        x,
        flux,
        linewidth=1.1,
        color="#2563eb",
        label="Flux",
    )

    ax.axhline(
        0,
        linewidth=0.7,
        color="gray",
        alpha=0.5,
    )

    ax.set_xlabel("Index binu")
    ax.set_ylabel("Flux")
    ax.set_title(title, fontsize=10)

    ax.grid(alpha=0.2)

    mask_axis = ax.twinx()

    mask_axis.step(
        x,
        coverage,
        where="mid",
        color="#d97706",
        linewidth=0.9,
        alpha=0.65,
        label="Pokryti",
    )

    if (
        float(coverage.min()) >= -0.01
        and float(coverage.max()) <= 1.01
    ):
        mask_axis.set_ylim(-0.1, 1.1)

    mask_axis.set_ylabel(
        "Pokryti (kanal 1)",
        fontsize=8,
        color="#d97706",
    )

    mask_axis.tick_params(
        axis="y",
        labelsize=8,
    )


def create_figure(
    examples,
    validation,
    title,
    destination,
):
    """Sestaví srovnávací grafy z vybraných validačních příkladů."""
    count = len(examples)

    fig, axes = plt.subplots(
        count,
        2,
        figsize=(16, count * 3.2),
        constrained_layout=True,
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    for row_number, example in enumerate(examples):
        position = example["position"]

        uid = example["signal_uid"]
        tic = example["tic_key"]

        probability = example["tuned_probability"]

        local_view = validation["local"][position]
        global_view = validation["global"][position]

        plot_view(
            axes[row_number, 0],
            local_view,
            (
                f"Local | TIC {tic} | "
                f"p(planet) = {probability:.3f}"
            ),
        )

        plot_view(
            axes[row_number, 1],
            global_view,
            f"Global | signal {uid}",
        )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        destination,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure saved:", destination)


def write_report(false_positives, false_negatives):
    """Uloží stručnou dokumentaci vizualizovaných chyb."""
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Model B - Visual Inspection of Persistent Errors",
        "",
        "## Method",
        "",
        "- Only validation signals were selected.",
        "- A persistent error occurred in all three tuned-model seeds.",
        "- FP examples are ranked by decreasing mean predicted planet probability.",
        "- FN examples are ranked by increasing mean predicted planet probability.",
        "- Four examples of each error type are displayed.",
        "- Local and global flux views come from the existing CNN NPZ dataset.",
        "- The second NPZ channel is displayed as coverage information.",
        "- Horizontal coordinates are bin indices, not physical time.",
        "- No held-out test signals were plotted.",
        "",
        "## Persistent false positives",
        "",
        "These signals are labelled false positive,",
        "but tuned Mantis consistently predicts planet.",
        "",
        "| Signal UID | TIC | Mean P(planet) | Original errors / 3 |",
        "|---|---|---:|---:|",
    ]

    for item in false_positives:
        lines.append(
            f"| {item['signal_uid']} "
            f"| {item['tic_key']} "
            f"| {item['tuned_probability']:.4f} "
            f"| {item['original_errors']} |"
        )

    lines.extend([
        "",
        "![Persistent false positives]"
        "(figures/model_b_persistent_false_positives.png)",
        "",
        "## Persistent false negatives",
        "",
        "These signals are labelled confirmed planets,",
        "but tuned Mantis consistently predicts non-planet.",
        "",
        "| Signal UID | TIC | Mean P(planet) | Original errors / 3 |",
        "|---|---|---:|---:|",
    ])

    for item in false_negatives:
        lines.append(
            f"| {item['signal_uid']} "
            f"| {item['tic_key']} "
            f"| {item['tuned_probability']:.4f} "
            f"| {item['original_errors']} |"
        )

    lines.extend([
        "",
        "![Persistent false negatives]"
        "(figures/model_b_persistent_false_negatives.png)",
        "",
        "## Interpretation",
        "",
        "The figures identify difficult validation examples.",
        "They do not establish the physical cause of each error.",
        "Possible explanations require further inspection of",
        "metadata, quality flags, coverage, and source light curves.",
        "",
        "The plotted flux is the preprocessed representation,",
        "not the unmodified original TESS FITS flux.",
        "",
    ])

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Report saved:", REPORT)


def main():
    """Vygeneruje grafy obtížných validačních příkladů."""
    print("=" * 70)
    print("MODEL B - PERSISTENT ERROR VISUALIZATION")
    print("=" * 70)

    validation = load_validation()

    examples = load_error_details(
        validation
    )

    false_positives, false_negatives = (
        select_examples(examples)
    )

    print("\nSelected false positives:")

    for item in false_positives:
        print(
            item["signal_uid"],
            "TIC:",
            item["tic_key"],
            "P(planet):",
            f"{item['tuned_probability']:.4f}",
        )

    print("\nSelected false negatives:")

    for item in false_negatives:
        print(
            item["signal_uid"],
            "TIC:",
            item["tic_key"],
            "P(planet):",
            f"{item['tuned_probability']:.4f}",
        )

    create_figure(
        false_positives,
        validation,
        "TESS - Persistent False Positives",
        FIGURES / "model_b_persistent_false_positives.png",
    )

    create_figure(
        false_negatives,
        validation,
        "TESS - Persistent False Negatives",
        FIGURES / "model_b_persistent_false_negatives.png",
    )

    write_report(
        false_positives,
        false_negatives,
    )

    print("\nTEST SET NOT EVALUATED")
    print("VISUAL ERROR ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
