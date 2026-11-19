
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "full_dataset"
OUTPUT.mkdir(exist_ok=True)

SIGNALS = BASE / "signals.csv"
REPORT = OUTPUT / "training_ready_audit.csv"
CROSSTAB = OUTPUT / "training_ready_labels.csv"


def main():
    signals = pd.read_csv(
        SIGNALS,
        low_memory=False
    )

    required = [
        "training_ready_minimum",
        "signal_label",
        "period_days",
        "epoch_bjd",
        "duration_hours",
    ]

    missing = [
        col for col in required
        if col not in signals.columns
    ]

    if missing:
        raise ValueError(
            f"Chybejici sloupce: {missing}"
        )

    print("=" * 70)
    print("AUDIT TRAINING_READY_MINIMUM")
    print("=" * 70)

    print("\n========== HODNOTY PRIZNAKU ==========")

    print(
        signals["training_ready_minimum"]
        .astype("string")
        .fillna("<missing>")
        .value_counts(dropna=False)
        .to_string()
    )

    ready = (
        signals["training_ready_minimum"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
    )

    signals["ready"] = ready

    print("\n========== READY VS SIGNAL_LABEL ==========")

    table = pd.crosstab(
        signals["signal_label"].fillna("<missing>"),
        signals["ready"],
        margins=True
    )

    print(table.to_string())

    table.to_csv(
        CROSSTAB,
        encoding="utf-8-sig"
    )

    for col in [
        "period_days",
        "epoch_bjd",
        "duration_hours",
    ]:
        signals[col] = pd.to_numeric(
            signals[col],
            errors="coerce"
        )

    period = signals["period_days"]
    epoch = signals["epoch_bjd"]
    duration = signals["duration_hours"]

    conditions = {
        "period_missing_or_nonfinite": (
            ~np.isfinite(period)
        ),
        "period_nonpositive": (
            np.isfinite(period) & (period <= 0)
        ),
        "epoch_missing_or_nonfinite": (
            ~np.isfinite(epoch)
        ),
        "duration_missing_or_nonfinite": (
            ~np.isfinite(duration)
        ),
        "duration_nonpositive": (
            np.isfinite(duration) & (duration <= 0)
        ),
        "duration_not_shorter_than_period": (
            np.isfinite(period)
            & np.isfinite(duration)
            & (duration >= period * 24)
        ),
        "invalid_label": (
            ~signals["signal_label"].isin([
                "candidate",
                "confirmed",
                "false_positive",
            ])
        ),
    }

    rows = []

    for name, condition in conditions.items():
        rows.append({
            "condition": name,
            "all_signals": int(condition.sum()),
            "ready_true": int((condition & ready).sum()),
            "ready_false": int((condition & ~ready).sum()),
        })

    report = pd.DataFrame(rows)

    print("\n========== PROBLEMY V METADATECH ==========")
    print(report.to_string(index=False))

    report.to_csv(
        REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== SOUHRN ==========")
    print("Celkem signalu:", len(signals))
    print("Ready TRUE:", int(ready.sum()))
    print("Ready FALSE:", int((~ready).sum()))

    print("\nReporty:")
    print(REPORT)
    print(CROSSTAB)

    print("\nAUDIT DOKONCEN.")


if __name__ == "__main__":
    main()
