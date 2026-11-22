"""Kontroluje zastoupení tříd v trénovacím manifestu."""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
PATH = BASE / "full_dataset" / "training_manifest.csv"


def main():
    """Načte trénovací manifest a vypíše rozložení tříd."""
    if not PATH.exists():
        raise FileNotFoundError(
            f"Training manifest nebyl nalezen: {PATH}"
        )

    df = pd.read_csv(PATH, low_memory=False)

    required = [
        "signal_label",
        "signal_source",
        "source_pipeline",
        "tic_key",
        "period_days",
        "duration_hours",
        "audit_ok_fits_count",
        "preliminary_ready",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Chybejici sloupce: {missing}")

    ready = (
        df["preliminary_ready"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )

    df = df.loc[ready].copy()

    print("=" * 75)
    print("ANALYZA TRID A ZDROJU SIGNALU TESS")
    print("=" * 75)

    print("\n1. ROZLOZENI TRID")
    print(df["signal_label"].value_counts().to_string())

    print("\n2. TRIDY PODLE ZDROJE (TOI / CTOI / TCE)")
    print(
        pd.crosstab(
            df["signal_source"],
            df["signal_label"],
        ).to_string()
    )

    print("\n3. TRIDY PODLE PIPELINE")
    print(
        pd.crosstab(
            df["source_pipeline"].fillna("missing"),
            df["signal_label"],
        ).to_string()
    )

    print("\n4. UNIKATNI HVEZDY PODLE TRIDY")
    print(
        df.groupby("signal_label")["tic_key"]
        .nunique()
        .to_string()
    )

    labeled = df.loc[
        df["signal_label"].isin(
            ["confirmed", "false_positive"]
        )
    ].copy()

    confirmed_tics = set(
        labeled.loc[
            labeled["signal_label"] == "confirmed",
            "tic_key",
        ].dropna()
    )

    false_positive_tics = set(
        labeled.loc[
            labeled["signal_label"] == "false_positive",
            "tic_key",
        ].dropna()
    )

    print("\n5. BINARNI KLASIFIKACE")
    print("Oznacene signaly:", len(labeled))
    print("Unikatni hvezdy:", labeled["tic_key"].nunique())
    print(
        "Hvezdy s obema typy signalu:",
        len(confirmed_tics & false_positive_tics),
    )

    print("\n6. POCET FITS NA SIGNAL")
    print(
        labeled["audit_ok_fits_count"]
        .describe()
        .to_string()
    )

    print("\n7. PERIODA PODLE TRIDY (DNY)")
    print(
        labeled.groupby("signal_label")["period_days"]
        .agg(["count", "median", "min", "max"])
        .to_string()
    )

    print("\n8. DELKA TRANZITU PODLE TRIDY (HODINY)")
    print(
        labeled.groupby("signal_label")["duration_hours"]
        .agg(["count", "median", "min", "max"])
        .to_string()
    )

    print("\nKONTROLA DOKONCENA")


if __name__ == "__main__":
    main()
