
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

INPUT = DATA / "transit_coverage_detail.csv"
OUTPUT = DATA / "transit_coverage_flags.csv"


def main():
    print("=" * 70)
    print("TESS - ZAVERECNA KONTROLA POKRYTI")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Soubor nebyl nalezen: {INPUT}"
        )

    df = pd.read_csv(INPUT, low_memory=False)

    required = [
        "signal_uid",
        "tic_key",
        "signal_label",
        "period_days",
        "duration_hours",
        "fits_processed",
        "points_in_transit",
        "points_near_transit",
        "coverage_pass",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Chybejici sloupce: {missing}"
        )

    df["coverage_pass"] = (
        df["coverage_pass"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )

    numeric_columns = [
        "period_days",
        "duration_hours",
        "fits_processed",
        "points_in_transit",
        "points_near_transit",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Podil delky tranzitu na cele obezne periode.
    df["duration_period_ratio"] = (
        df["duration_hours"]
        / (df["period_days"] * 24.0)
    )

    df["invalid_period"] = (
        ~np.isfinite(df["period_days"])
        | (df["period_days"] <= 0)
    )

    df["invalid_duration"] = (
        ~np.isfinite(df["duration_hours"])
        | (df["duration_hours"] <= 0)
    )

    df["unusual_duration_ratio"] = (
        df["duration_period_ratio"] >= 0.25
    ).fillna(False)

    df["duration_exceeds_period"] = (
        df["duration_period_ratio"] >= 1.0
    ).fillna(False)

    df["coverage_failed"] = ~df["coverage_pass"]

    flags = df.loc[
        df["coverage_failed"]
        | df["invalid_period"]
        | df["invalid_duration"]
        | df["unusual_duration_ratio"]
    ].copy()

    print("\n1. ZAKLADNI SOUHRN")
    print("Celkem signalu:", len(df))
    print("Pokryti proslo:", int(df["coverage_pass"].sum()))
    print("Pokryti neproslo:", int(df["coverage_failed"].sum()))

    print("\n2. SIGNALY BEZ DOSTATECNEHO POKRYTI")

    failed = df.loc[df["coverage_failed"]]

    columns = [
        "signal_uid",
        "tic_key",
        "signal_label",
        "period_days",
        "duration_hours",
        "fits_processed",
        "points_in_transit",
        "points_near_transit",
    ]

    if failed.empty:
        print("Zadne.")
    else:
        print(
            failed[columns].to_string(index=False)
        )

    print("\n3. KONTROLA PERIOD A DELEK TRANZITU")

    print(
        "Neplatna perioda:",
        int(df["invalid_period"].sum()),
    )

    print(
        "Neplatna delka tranzitu:",
        int(df["invalid_duration"].sum()),
    )

    print(
        "Tranzit >= 25 % periody:",
        int(df["unusual_duration_ratio"].sum()),
    )

    print(
        "Tranzit >= 100 % periody:",
        int(df["duration_exceeds_period"].sum()),
    )

    print("\n4. PODEZRELE POMERY PODLE TRIDY")

    print(
        pd.crosstab(
            df["signal_label"],
            df["unusual_duration_ratio"],
        ).to_string()
    )

    print("\n5. STATISTIKA PERIOD A DELKY TRANZITU")

    print(
        df[
            [
                "period_days",
                "duration_hours",
                "duration_period_ratio",
            ]
        ].describe(
            percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
        ).to_string()
    )

    flags.to_csv(OUTPUT, index=False)

    print("\nPocet signalu k provereni:", len(flags))
    print("Report:", OUTPUT)
    print("\nHOTOVO")


if __name__ == "__main__":
    main()
