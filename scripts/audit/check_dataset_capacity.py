
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent

SIGNALS = BASE / "signals.csv"
LIGHTCURVES = BASE / "lightcurves.csv"
MANIFEST = (
    BASE / "TESS_sample_150"
    / "tess_sample_150_manifest.csv"
)

OUTPUT = BASE / "dataset_capacity_report.csv"

LABELS = [
    "candidate",
    "confirmed",
    "false_positive",
]


def normalize_id(value):
    if pd.isna(value):
        return ""

    result = str(value).strip()

    if result.startswith("TIC_"):
        result = result[4:]

    if result.endswith(".0"):
        result = result[:-2]

    return result


def main():
    print("=" * 65)
    print("KONTROLA KAPACITY DATASETU TESS")
    print("=" * 65)

    for path in [SIGNALS, LIGHTCURVES, MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(
                f"Soubor nebyl nalezen: {path}"
            )

    signals = pd.read_csv(
        SIGNALS, low_memory=False
    )

    lightcurves = pd.read_csv(
        LIGHTCURVES, low_memory=False
    )

    manifest = pd.read_csv(
        MANIFEST, low_memory=False
    )

    required = {
        "tic_id",
        "signal_uid",
        "signal_label",
        "period_days",
        "epoch_bjd",
        "duration_hours",
        "training_ready_minimum",
        "n_fits_for_this_tic",
    }

    missing = required - set(signals.columns)

    if missing:
        raise ValueError(
            f"Chybejici sloupce: {sorted(missing)}"
        )

    signals["tic_id"] = (
        signals["tic_id"].map(normalize_id)
    )

    signals["signal_uid"] = (
        signals["signal_uid"].astype(str)
    )

    manifest["tic_id"] = (
        manifest["tic_id"].map(normalize_id)
    )

    for column in [
        "period_days",
        "epoch_bjd",
        "duration_hours",
        "n_fits_for_this_tic",
    ]:
        signals[column] = pd.to_numeric(
            signals[column],
            errors="coerce"
        )

    ready = (
        signals["training_ready_minimum"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    valid_parameters = (
        np.isfinite(signals["period_days"])
        & (signals["period_days"] > 0)
        & np.isfinite(signals["epoch_bjd"])
        & np.isfinite(signals["duration_hours"])
        & (signals["duration_hours"] > 0)
        & (
            signals["duration_hours"]
            < signals["period_days"] * 24
        )
    )

    has_fits = (
        signals["n_fits_for_this_tic"] > 0
    )

    known_label = (
        signals["signal_label"].isin(LABELS)
    )

    not_pilot = (
        ~signals["tic_id"].isin(
            set(manifest["tic_id"])
        )
    )

    eligible = signals[
        ready
        & valid_parameters
        & has_fits
        & known_label
        & not_pilot
        & (signals["tic_id"] != "")
    ].copy()

    # Pro prvni odhad chceme nejvyse jeden
    # signal na jednu hvezdu.
    #
    # Pokud ma hvezda vice signalu, muze mit
    # i vice ruznych klasifikaci.
    # Takove hvezdy zatim odlozime.
    labels_per_tic = (
        eligible.groupby("tic_id")["signal_label"]
        .nunique()
    )

    conflicting_tics = set(
        labels_per_tic[
            labels_per_tic > 1
        ].index
    )

    eligible = eligible[
        ~eligible["tic_id"].isin(conflicting_tics)
    ].copy()

    # Preferujeme jeden zaznam na TIC.
    # Toto je pouze odhad kapacity, nikoli
    # definitivni vyber trenovacich dat.
    eligible = eligible.sort_values(
        ["tic_id", "signal_uid"]
    ).drop_duplicates(
        subset=["tic_id"],
        keep="first"
    )

    counts = Counter(
        eligible["signal_label"]
    )

    print(f"\nSignals celkem: {len(signals)}")
    print(
        f"Lightcurves metadata: "
        f"{len(lightcurves)}"
    )
    print(
        f"Pilotni vzorek: "
        f"{len(manifest)}"
    )

    print(
        "\nZaznamy splnujici zakladni "
        "metadata kontroly:"
    )

    print(
        int(
            (
                ready
                & valid_parameters
                & has_fits
                & known_label
            ).sum()
        )
    )

    print(
        "\nHvezdy s konfliktnimi tridami "
        "mezi predbezne vhodnymi signaly:"
    )
    print(len(conflicting_tics))

    print(
        "\nDalsi potencialne vhodne "
        "unikatni hvezdy:"
    )

    for label in LABELS:
        print(
            f"{label:16s}: {counts[label]}"
        )

    minimum = min(
        counts[label] for label in LABELS
    )

    print(
        "\nMaximalni vyvazeny pocet "
        "novych signalu podle metadat:"
    )
    print(minimum * 3)

    print(
        "\nPoznamka: Tento pocet "
        "NEPOTVRZUJE dostupnost ani "
        "kvalitu konkretnich FITS souboru."
    )

    summary = pd.DataFrame([
        {
            "signal_label": label,
            "available_unique_tics": counts[label],
            "target_new_signals": 450,
            "enough_for_target": (
                counts[label] >= 450
            ),
        }
        for label in LABELS
    ])

    summary.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSouhrn:")
    print(summary.to_string(index=False))

    print(f"\nReport: {OUTPUT}")
    print("\nKontrola dokoncena.")


if __name__ == "__main__":
    main()
