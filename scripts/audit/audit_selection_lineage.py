
"""
Read-only audit of how binary TESS signals become the 1,918 CNN inputs.

Execute from the project root:
    python -u scripts/audit/audit_selection_lineage.py

Reads existing CSV/NPZ artifacts. Never rebuilds datasets or runs test models.
Writes only docs/dataset_selection_lineage.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "full_dataset"
REPORT = ROOT / "docs" / "dataset_selection_lineage.md"
LABELS = ("confirmed", "false_positive")


def check(condition, message):
    """Ukončí audit při porušení požadované integrity dat."""
    if not condition:
        raise RuntimeError(message)


def read_table(path, required):
    """Načte CSV a ověří přítomnost požadovaných sloupců."""
    check(path.is_file(), f"Chybi vstupni soubor: {path}")
    frame = pd.read_csv(
        path,
        dtype={"signal_uid": str},
        low_memory=False,
    )
    missing = set(required) - set(frame.columns)
    check(
        not missing,
        f"{path.name}: chybi sloupce {sorted(missing)}",
    )
    frame["signal_uid"] = (
        frame["signal_uid"].astype(str).str.strip()
    )
    check(
        not frame["signal_uid"].duplicated().any(),
        f"{path.name}: duplicitni signal_uid",
    )
    check(
        not frame["signal_uid"].isin(["", "nan", "None"]).any(),
        f"{path.name}: prazdny signal_uid",
    )
    return frame


def is_true(column):
    """Sjednotí textové příznaky logických hodnot."""
    return (
        column.astype("string")
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
        .fillna(False)
    )


def counts(ids, source_labels):
    """Vrátí počty signálů podle katalogové třídy."""
    labels = source_labels.loc[list(ids)]
    return (
        len(ids),
        int((labels == "confirmed").sum()),
        int((labels == "false_positive").sum()),
    )


def main():
    """Zrekonstruuje a ověří jednotlivé kroky výběru signálů."""
    print("=" * 72)
    print("TESS - AUDIT VYBERU SIGNALU")
    print("=" * 72)

    signals = read_table(
        ROOT / "signals.csv",
        ("signal_uid", "signal_label"),
    )

    coverage = read_table(
        DATA / "transit_coverage_detail.csv",
        (
            "signal_uid",
            "signal_label",
            "coverage_pass",
            "period_days",
            "duration_hours",
        ),
    )

    qc = read_table(
        DATA / "full_cnn_qc.csv",
        ("signal_uid", "status", "signal_label"),
    )

    manifest = read_table(
        DATA / "training_manifest.csv",
        ("signal_uid", "preliminary_ready", "signal_label"),
    )

    npz_path = DATA / "full_cnn_views.npz"

    check(
        npz_path.is_file(),
        f"Chybi dataset: {npz_path}",
    )

    signals["signal_label"] = (
        signals["signal_label"].astype(str).str.strip()
    )

    source = signals.set_index("signal_uid")["signal_label"]

    binary_ids = set(
        signals.loc[
            signals["signal_label"].isin(LABELS),
            "signal_uid",
        ]
    )

    check(
        len(binary_ids) == 1978,
        f"Neocekavany pocet binarnich signalu: {len(binary_ids)}",
    )

    print("\nKontroluji vazby na puvodni katalog...")

    for name, frame in (
        ("coverage", coverage),
        ("QC", qc),
        ("manifest", manifest),
    ):
        check(
            set(frame["signal_uid"]).issubset(set(source.index)),
            f"{name}: signaly mimo katalog",
        )

        subset = frame.loc[
            frame["signal_uid"].isin(binary_ids)
        ].copy()

        expected = subset["signal_uid"].map(source)

        check(
            (
                subset["signal_label"].astype(str).str.strip()
                == expected
            ).all(),
            f"{name}: nesouhlasi katalogove stitky",
        )

    print("Katalogove stitky: OK")

    covered = coverage.loc[
        coverage["signal_uid"].isin(binary_ids)
        & is_true(coverage["coverage_pass"])
        & coverage["signal_label"].isin(LABELS)
    ].copy()

    covered_ids = set(covered["signal_uid"])

    period = pd.to_numeric(
        covered["period_days"],
        errors="coerce",
    )

    duration = pd.to_numeric(
        covered["duration_hours"],
        errors="coerce",
    )

    unusual = (
        duration / (24 * period)
    ) >= 0.25

    selected_ids = set(
        covered.loc[~unusual, "signal_uid"]
    )

    qc_ids = set(qc["signal_uid"])

    check(
        qc_ids == selected_ids,
        (
            "QC neobsahuje presne vybrane signaly. "
            f"Chybi v QC: {len(selected_ids - qc_ids)}; "
            f"navic v QC: {len(qc_ids - selected_ids)}"
        ),
    )

    passed_ids = set(
        qc.loc[
            qc["status"].astype(str).str.strip().eq("ok"),
            "signal_uid",
        ]
    )

    print("Nacitam finalni NPZ metadata...")

    with np.load(npz_path, allow_pickle=False) as dataset:
        uid = dataset["signal_uid"].astype(str)
        labels = dataset["y"].astype(int)

    check(
        len(uid) == len(set(uid)),
        "NPZ obsahuje duplicitni signal_uid",
    )

    final_ids = set(uid)

    check(
        final_ids == passed_ids,
        (
            "NPZ a QC nesouhlasi. "
            f"NPZ mimo QC: {len(final_ids - passed_ids)}; "
            f"QC mimo NPZ: {len(passed_ids - final_ids)}"
        ),
    )

    check(
        len(uid) == 1918,
        f"Neocekavany pocet NPZ signalu: {len(uid)}",
    )

    expected_y = np.asarray([
        1 if source.loc[signal] == "confirmed" else 0
        for signal in uid
    ])

    check(
        np.array_equal(labels, expected_y),
        "NPZ stitky nesouhlasi s katalogem",
    )

    stages = [
        ("Katalogove oznacene signaly", binary_ids),
        ("Pokryti tranzitu: coverage_pass", covered_ids),
        ("Delka tranzitu / perioda < 0.25", selected_ids),
        ("Uspesne pripraveny pohled", passed_ids),
    ]

    print("\n========== POSTUP VYBERU ==========")

    print(
        f"{'Krok':43s} "
        f"{'Celkem':>7s} "
        f"{'Planet':>8s} "
        f"{'FP':>7s} "
        f"{'Ubytek':>8s}"
    )

    table_rows = []
    previous = None

    for name, ids in stages:
        total, positives, negatives = counts(ids, source)

        lost = (
            0 if previous is None
            else len(previous - ids)
        )

        if previous is not None:
            check(
                ids.issubset(previous),
                f"{name}: neocekavany vyber signalu",
            )

        print(
            f"{name:43s} "
            f"{total:7d} "
            f"{positives:8d} "
            f"{negatives:7d} "
            f"{lost:8d}"
        )

        table_rows.append(
            (name, total, positives, negatives, lost)
        )

        previous = ids

    print("\n========== DUVODY NEUSPESNE PRIPRAVY ==========")

    rejected = qc.loc[
        ~qc["status"].astype(str).str.strip().eq("ok")
    ]

    rejection_counts = (
        rejected["status"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    if len(rejection_counts):
        for reason, number in rejection_counts.items():
            print(f"{reason}: {number}")
    else:
        print("Zadne")

    ready = manifest.loc[
        manifest["signal_uid"].isin(binary_ids)
    ]

    ready_ids = set(
        ready.loc[
            is_true(ready["preliminary_ready"]),
            "signal_uid",
        ]
    )

    print("\n========== KONTROLA MANIFESTU ==========")

    print(
        "Preliminary ready - binarni signaly:",
        len(ready_ids),
    )

    print(
        "Finalni signaly mimo preliminary ready:",
        len(final_ids - ready_ids),
    )

    excluded = binary_ids - final_ids

    check(
        len(excluded) == 60,
        "Nesouhlasi celkovy pocet vyrazenych signalu",
    )

    print("\n========== FINALNI SOUHRN ==========")

    print("Katalogove binarni signaly:", len(binary_ids))
    print("Finalni dataset:", len(final_ids))
    print("Celkem vyrazeno:", len(excluded))
    print("Shoda QC a NPZ: PASS")
    print("Shoda katalogovych stitku: PASS")
    print("Testovaci predikce provedeny: NE")

    lines = [
        "# TESS - Data Selection Lineage",
        "",
        "Audit of existing experiment artifacts.",
        "No FITS data, models or test metrics were regenerated.",
        "",
        "## Signal selection",
        "",
        "| Stage | Signals | Confirmed | False positive | Removed |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, total, positives, negatives, lost in table_rows:
        lines.append(
            f"| {name} | {total} | {positives} "
            f"| {negatives} | {lost} |"
        )

    lines.extend([
        "",
        "## Final exclusions",
        "",
        f"- Original binary signals: {len(binary_ids)}",
        f"- Final prepared signals: {len(final_ids)}",
        f"- Excluded signals: {len(excluded)}",
        "",
        "## Preparation rejection reasons",
        "",
    ])

    for reason, number in rejection_counts.items():
        lines.append(
            f"- `{reason}`: {number}"
        )

    lines.extend([
        "",
        "## Integrity",
        "",
        "- Final NPZ identifiers match QC rows with status `ok`.",
        "- Binary NPZ labels match original catalogue metadata.",
        f"- Preliminary-ready binary signals: {len(ready_ids)}",
        (
            "- Final signals absent from preliminary-ready subset: "
            f"{len(final_ids - ready_ids)}"
        ),
        "- Held-out test predictions were not regenerated.",
        "",
    ])

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print("\nReport:", REPORT)
    print("AUDIT DOKONCEN")


if __name__ == "__main__":
    main()
