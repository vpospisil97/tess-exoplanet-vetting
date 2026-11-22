"""Kontroluje výběr katalogových signálů a dostupných světelných křivek."""


from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "full_dataset"
OUTPUT.mkdir(exist_ok=True)


def normalize_tic(value):
    """Sjednotí identifikátor TIC pro propojení datových tabulek."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.startswith("TIC_"):
        text = text[4:]

    if text.endswith(".0"):
        text = text[:-2]

    return text


def safe_path(value):
    """Ověří, že katalogová cesta neuniká mimo pracovní adresář."""
    path = Path(str(value).replace("\\", "/"))

    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.suffix.lower() == ".fits"
        and len(path.parts) == 2
    )


def main():
    """Zkontroluje výběr katalogových signálů a odpovídajících FITS."""
    signals = pd.read_csv(
        BASE / "signals.csv",
        low_memory=False
    )

    lightcurves = pd.read_csv(
        BASE / "lightcurves.csv",
        low_memory=False
    )

    signals["tic_id"] = signals["tic_id"].map(
        normalize_tic
    )

    lightcurves["tic_id"] = lightcurves["tic_id"].map(
        normalize_tic
    )

    for column in [
        "period_days",
        "epoch_bjd",
        "duration_hours",
    ]:
        signals[column] = pd.to_numeric(
            signals[column],
            errors="coerce"
        )

    period = signals["period_days"]
    epoch = signals["epoch_bjd"]
    duration = signals["duration_hours"]

    checks = {
        "training_ready_minimum": (
            signals["training_ready_minimum"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "true", "1"])
        ),
        "valid_label": signals["signal_label"].isin([
            "candidate",
            "confirmed",
            "false_positive",
        ]),
        "valid_period": (
            np.isfinite(period) & (period > 0)
        ),
        "valid_epoch": np.isfinite(epoch),
        "valid_duration": (
            np.isfinite(duration) & (duration > 0)
        ),
        "duration_shorter_than_period": (
            np.isfinite(period)
            & np.isfinite(duration)
            & (duration < period * 24)
        ),
        "valid_tic": signals["tic_id"] != "",
    }

    rows = []
    remaining = pd.Series(
        True,
        index=signals.index
    )

    for name, condition in checks.items():
        before = int(remaining.sum())
        failed_individually = int((~condition).sum())

        remaining = remaining & condition

        after = int(remaining.sum())

        rows.append({
            "step": name,
            "before": before,
            "removed_at_step": before - after,
            "after": after,
            "failed_individually": failed_individually,
        })

    print("\n========== AUDIT SIGNALU ==========")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n========== AUDIT FITS METADAT ==========")

    original_lc = len(lightcurves)

    lc = lightcurves[
        lightcurves["tic_id"] != ""
    ].copy()

    after_tic = len(lc)

    lc["hf_path"] = lc["hf_path"].astype(str)

    lc = lc[lc["hf_path"].map(safe_path)].copy()
    after_path = len(lc)

    conflicts = (
        lc.groupby("hf_path")["tic_id"]
        .nunique()
    )

    conflict_count = int((conflicts > 1).sum())

    if conflict_count:
        raise ValueError(
            f"Konfliktnich FITS cest: {conflict_count}"
        )

    lc = lc.drop_duplicates(subset=["hf_path"])
    after_dedup = len(lc)

    eligible = signals[remaining].copy()
    before_link = len(eligible)

    eligible = eligible[
        eligible["tic_id"].isin(set(lc["tic_id"]))
    ].copy()

    after_link = len(eligible)

    selected_tics = set(eligible["tic_id"])

    selected_lc = lc[
        lc["tic_id"].isin(selected_tics)
    ].copy()

    lc_rows = [
        ("original_lightcurves", original_lc),
        ("after_valid_tic", after_tic),
        ("after_safe_path", after_path),
        ("after_path_deduplication", after_dedup),
        ("selected_fits", len(selected_lc)),
    ]

    print(pd.DataFrame(
        lc_rows,
        columns=["step", "count"]
    ).to_string(index=False))

    print("\n========== KONTROLA VYSLEDKU ==========")
    print("Signaly pred propojenim:", before_link)
    print("Signaly po propojeni:", after_link)
    print("Unikatni TIC:", eligible["tic_id"].nunique())
    print("Vybrane FITS:", len(selected_lc))
    print("Konfliktni FITS cesty:", conflict_count)

    signal_report = pd.DataFrame(rows)
    signal_report.loc[len(signal_report)] = {
        "step": "tic_has_lightcurve",
        "before": before_link,
        "removed_at_step": before_link - after_link,
        "after": after_link,
        "failed_individually": before_link - after_link,
    }

    signal_report.to_csv(
        OUTPUT / "selection_audit_signals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        lc_rows,
        columns=["step", "count"]
    ).to_csv(
        OUTPUT / "selection_audit_lightcurves.csv",
        index=False,
        encoding="utf-8-sig"
    )

    assert len(eligible) == 8437, (
        "Pocet signalu neodpovida puvodnimu manifestu."
    )

    assert len(selected_lc) == 30864, (
        "Pocet FITS neodpovida puvodnimu manifestu."
    )

    print("\nAUDIT USPESNY.")
    print("Reporty ulozeny do full_dataset.")


if __name__ == "__main__":
    main()
