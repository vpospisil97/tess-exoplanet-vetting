"""
Vytváří závěrečné grafy testovacích výsledků TESS z uloženého reportu.

Skript pouze čte Markdown tabulky, ověřuje uvedené metriky
a zapisuje grafy. Nikdy nespouští trénování ani evaluaci modelů.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "final_test_results.md"
OUTPUT = ROOT / "docs" / "figures"

SEEDS = [42, 123, 2026]

MODEL_NAMES = {
    "Logistic": "Logistic regression",
    "CNN": "Custom CNN",
    "Mantis": "Mantis-8M + MLP",
}

METRICS = {
    "AP": "Average Precision",
    "F1": "F1-score",
}

TEST_SIZE = 290
TEST_POSITIVE = 195
TEST_NEGATIVE = 95


def parse_table(section_name):
    """Načte požadovanou tabulku z výsledkového Markdown reportu.

    Vrací seznam záznamů, kde názvy sloupců tvoří klíče slovníků.
    Při chybějící sekci nebo neplatném řádku vyvolá výjimku."""

    text = REPORT.read_text(encoding="utf-8")
    lines = text.splitlines()

    start = None

    for index, line in enumerate(lines):
        if line.startswith(f"## {section_name}"):
            start = index + 1
            break

    if start is None:
        raise RuntimeError(
            f"Section not found: {section_name}"
        )

    rows = []

    for line in lines[start:]:
        if line.startswith("## "):
            break

        if not line.startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        if all(
            cell and set(cell) <= set("-: ")
            for cell in cells
        ):
            continue

        rows.append(cells)

    if len(rows) < 2:
        raise RuntimeError(
            f"No table data in: {section_name}"
        )

    header = rows[0]
    result = []

    for cells in rows[1:]:
        if len(cells) != len(header):
            raise RuntimeError(
                f"Invalid table row in: {section_name}"
            )

        result.append(dict(zip(header, cells)))

    return result


def numeric(value):
    """Převede text metriky na číselnou střední hodnotu.

    Pokud je uvedeno „průměr ± odchylka“, použije pouze průměr."""

    return float(
        str(value).split("±")[0].strip()
    )


def get_summary():
    """Načte souhrnné AP a F1 jednotlivých modelů.

    Pro neuronové modely uchová také směrodatnou odchylku AP."""

    rows = parse_table("Performance")

    result = {}

    for row in rows:
        model = row["Model"]

        result[model] = {
            "AP": numeric(row["AP"]),
            "AP_SD": (
                float(row["AP"].split("±")[1].strip())
                if "±" in row["AP"]
                else 0.0
            ),
            "F1": numeric(row["F1"]),
        }

    if set(result) != set(MODEL_NAMES):
        raise RuntimeError(
            f"Unexpected summary models: {set(result)}"
        )

    return result


def get_runs():
    """Načte metriky jednotlivých seedů a ověří počty v maticích záměn.

    Každý běh musí obsahovat přesně 290 testovacích příkladů."""

    rows = parse_table("Per-run results")
    result = {}

    for row in rows:
        model = row["Model"]
        seed = row["Seed"]

        key = (
            (model, int(seed))
            if seed != "baseline"
            else (model, "baseline")
        )

        if key in result:
            raise RuntimeError(
                f"Duplicate result: {key}"
            )

        tn = int(row["TN"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tp = int(row["TP"])

        if tn + fp + fn + tp != TEST_SIZE:
            raise RuntimeError(
                f"Unexpected test sample count: {key}"
            )

        if tn + fp != TEST_NEGATIVE:
            raise RuntimeError(
                f"Unexpected negative count: {key}"
            )

        if fn + tp != TEST_POSITIVE:
            raise RuntimeError(
                f"Unexpected positive count: {key}"
            )

        result[key] = {
            "AP": float(row["AP"]),
            "F1": float(row["F1"]),
            "ROC": float(row["ROC-AUC"]),
        }

    expected = {
        ("Logistic", "baseline"),
        *[
            (model, seed)
            for model in ("CNN", "Mantis")
            for seed in SEEDS
        ],
    }

    if set(result) != expected:
        raise RuntimeError(
            "Missing or unexpected model/seed results."
        )

    return result


def validate_summary(summary, runs):
    """Porovná souhrnné metriky s hodnotami jednotlivých běhů.

    Kontroluje průměr a výběrovou směrodatnou odchylku AP
    a průměr F1 při zachování tolerance zaokrouhlení."""

    for model in ("CNN", "Mantis"):
        values = np.array([
            runs[(model, seed)]["AP"]
            for seed in SEEDS
        ])

        mean = float(values.mean())
        sd = float(values.std(ddof=1))

        if abs(mean - summary[model]["AP"]) > 0.0002:
            raise RuntimeError(
                f"AP mean mismatch for {model}"
            )

        if abs(sd - summary[model]["AP_SD"]) > 0.0002:
            raise RuntimeError(
                f"AP SD mismatch for {model}"
            )

        f1_prumer = float(np.mean([
            runs[(model, seed)]["F1"]
            for seed in SEEDS
        ]))

        if abs(f1_prumer - summary[model]["F1"]) > 0.0002:
            raise RuntimeError(
                f"F1 mean mismatch for {model}"
            )


def save_figure(fig, filename):
    """Uloží jeden graf jako vektorové PDF a rastrové PNG.

    Umístění obrázků odpovídá složce docs/figures.
    Po uložení uzavře instanci grafu."""

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    for extension in ("pdf", "png"):
        destination = OUTPUT / f"{filename}.{extension}"

        fig.savefig(
            destination,
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )

        print("Saved:", destination)

    plt.close(fig)


def plot_average_precision(summary):
    """Vykreslí finální srovnání AP tří klasifikačních přístupů.

    U CNN a Mantis znázorní výběrovou směrodatnou odchylku
    ze tří inicializací; baseline má jediný výsledek."""

    models = ["Logistic", "CNN", "Mantis"]

    values = [
        summary[model]["AP"]
        for model in models
    ]

    deviations = [
        summary[model]["AP_SD"]
        for model in models
    ]

    labels = [
        MODEL_NAMES[model]
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    positions = np.arange(len(models))

    bars = ax.bar(
        positions,
        values,
        yerr=deviations,
        capsize=7,
        width=0.58,
        alpha=0.85,
    )

    reference = TEST_POSITIVE / TEST_SIZE

    ax.axhline(
        reference,
        linestyle="--",
        linewidth=1.3,
        color="gray",
        label=(
            f"Positive class prevalence "
            f"({reference:.3f})"
        ),
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.022,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Precision")
    ax.set_title(
        "Held-out TESS test: model comparison",
        fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    fig.text(
        0.5,
        0.01,
        "290 test signals; CNN and Mantis: "
        "mean over 3 seeds, error bars = sample SD.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    save_figure(fig, "final_test_ap_comparison")


def plot_seed_metric(runs, metric):
    """Porovná výsledky neuronových modelů podle náhodných seedů.

    Seedy jsou identifikátory kategorií, nikoli číselná časová osa;
    pro všechny tři běhy proto používá rovnoměrné rozestupy."""

    fig, ax = plt.subplots(figsize=(9, 5.2))

    pozice = np.arange(len(SEEDS))

    for model in ("CNN", "Mantis"):
        values = [
            runs[(model, seed)][metric]
            for seed in SEEDS
        ]

        ax.plot(
            pozice,
            values,
            marker="o",
            markersize=8,
            linewidth=2,
            label=MODEL_NAMES[model],
        )

        for pozice_bodu, value in zip(pozice, values):
            ax.annotate(
                f"{value:.4f}",
                (pozice_bodu, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    baseline = runs[
        ("Logistic", "baseline")
    ][metric]

    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label=f"Logistic baseline ({baseline:.4f})",
    )

    all_values = [
        value[metric]
        for value in runs.values()
    ]

    minimum = max(
        0,
        min(all_values) - 0.04,
    )

    maximum = min(
        1,
        max(all_values) + 0.055,
    )

    ax.set_ylim(minimum, maximum)

    ax.set_xticks(pozice, [str(s) for s in SEEDS])
    ax.set_xlabel("Random seed")
    ax.set_ylabel(METRICS[metric])

    ax.set_title(
        f"Held-out TESS test: {METRICS[metric]} "
        "across seeds",
        fontsize=14,
    )

    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.text(
        0.5,
        0.01,
        "Same fixed TIC-disjoint test split for every seed. "
        "Y-axis zoomed.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    save_figure(
        fig,
        f"final_test_{metric.lower()}_by_seed",
    )


def main():
    """Ověří uložené finální metriky a znovu vygeneruje grafy.

    Neprovádí žádné trénování, predikci ani práci s testovacími FITS."""

    print("=" * 70)
    print("TESS - FINAL RESULTS FIGURES")
    print("=" * 70)

    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)

    summary = get_summary()
    runs = get_runs()

    validate_summary(summary, runs)

    print("\nSource report verified.")
    print("Models:", ", ".join(summary.keys()))
    print("Test signals:", TEST_SIZE)
    print("No training or inference will be performed.\n")

    plot_average_precision(summary)
    plot_seed_metric(runs, "AP")
    plot_seed_metric(runs, "F1")

    print("\nFIGURES GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
