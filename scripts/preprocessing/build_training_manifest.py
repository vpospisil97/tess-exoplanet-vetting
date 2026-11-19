from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

SIGNALS = BASE / "signals.csv"
LIGHTCURVES = BASE / "lightcurves.csv"
QUALITY = DATA / "fits_quality_full.csv"

OUT_SIGNALS = DATA / "training_manifest.csv"
OUT_LINKS = DATA / "training_fits_links.csv"
OUT_SUMMARY = DATA / "training_manifest_summary.csv"


def normalize_tic(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^TIC[_\s]*", "", regex=True, case=False)
    )


def normalize_path(series):
    return (
        series.astype("string")
        .str.replace("\\", "/", regex=False)
        .str.replace(r"^\./", "", regex=True)
        .str.replace(r"^full_dataset/fits/", "", regex=True)
    )


def main():
    print("=" * 75)
    print("TRAINING MANIFEST - TESS - VERZE 2")
    print("=" * 75)

    for path in [SIGNALS, LIGHTCURVES, QUALITY]:
        if not path.exists():
            raise FileNotFoundError(path)

    signals = pd.read_csv(SIGNALS, low_memory=False)
    lightcurves = pd.read_csv(LIGHTCURVES, low_memory=False)
    quality = pd.read_csv(QUALITY, low_memory=False)

    print("\nVSTUPY")
    print("Signals:", len(signals))
    print("Lightcurves:", len(lightcurves))
    print("FITS audit:", len(quality))

    required_signals = [
        "signal_uid", "tic_id", "signal_label",
        "period_days", "epoch_bjd",
        "duration_hours", "training_ready_minimum"
    ]

    required_lightcurves = ["hf_path", "tic_id", "sector"]
    required_quality = ["file", "status", "n_valid"]

    for name, df, required in [
        ("signals.csv", signals, required_signals),
        ("lightcurves.csv", lightcurves, required_lightcurves),
        ("fits_quality_full.csv", quality, required_quality),
    ]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name}: chybejici sloupce {missing}")

    if signals["signal_uid"].isna().any():
        raise ValueError("Prazdne signal_uid")

    if signals["signal_uid"].duplicated().any():
        raise ValueError("Duplicitni signal_uid")

    signals["tic_key"] = normalize_tic(signals["tic_id"])
    lightcurves["tic_key"] = normalize_tic(lightcurves["tic_id"])

    quality["fits_key"] = normalize_path(quality["file"])
    lightcurves["fits_key"] = normalize_path(lightcurves["hf_path"])

    # Podpora absolutnich cest v quality reportu.
    quality["fits_key"] = (
        quality["fits_key"]
        .str.replace(
            r"^.*?/full_dataset/fits/",
            "",
            regex=True,
        )
    )

    if quality["fits_key"].duplicated().any():
        raise ValueError("Duplicitni cesty v FITS auditu")

    if lightcurves["fits_key"].duplicated().any():
        raise ValueError("Duplicitni cesty v lightcurves.csv")

    for column in ["period_days", "epoch_bjd", "duration_hours"]:
        signals[column] = pd.to_numeric(
            signals[column], errors="coerce"
        )

    signals["metadata_valid"] = (
        np.isfinite(signals["period_days"])
        & signals["period_days"].gt(0)
        & np.isfinite(signals["epoch_bjd"])
        & signals["epoch_bjd"].gt(0)
        & np.isfinite(signals["duration_hours"])
        & signals["duration_hours"].gt(0)
    )

    signals["source_training_ready"] = (
        signals["training_ready_minimum"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("yes")
        .fillna(False)
    )

    quality["n_valid"] = pd.to_numeric(
        quality["n_valid"], errors="coerce"
    )

    quality["fits_audit_ok"] = (
        quality["status"].astype("string").str.lower().eq("ok")
        & quality["n_valid"].ge(100)
    )

    lightcurves = lightcurves.merge(
        quality[
            ["fits_key", "file", "status", "n_valid", "fits_audit_ok"]
        ],
        on="fits_key",
        how="left",
        validate="one_to_one",
        indicator="audit_match",
    )

    lightcurves["fits_found"] = (
        lightcurves["audit_match"].eq("both")
    )

    lightcurves["fits_audit_ok"] = (
        lightcurves["fits_audit_ok"].fillna(False).astype(bool)
    )

    files_by_tic = (
        lightcurves.groupby("tic_key")
        .agg(
            catalog_fits_count=("fits_key", "size"),
            downloaded_fits_count=("fits_found", "sum"),
            audit_ok_fits_count=("fits_audit_ok", "sum"),
        )
        .reset_index()
    )

    manifest = signals.merge(
        files_by_tic,
        on="tic_key",
        how="left",
        validate="many_to_one",
    )

    for column in [
        "catalog_fits_count",
        "downloaded_fits_count",
        "audit_ok_fits_count",
    ]:
        manifest[column] = (
            manifest[column].fillna(0).astype(int)
        )

    manifest["has_audit_ok_fits"] = (
        manifest["audit_ok_fits_count"] > 0
    )

    manifest["preliminary_ready"] = (
        manifest["source_training_ready"]
        & manifest["metadata_valid"]
        & manifest["has_audit_ok_fits"]
    )

    links = manifest[
        [
            "signal_uid", "tic_key", "signal_label",
            "period_days", "epoch_bjd", "duration_hours",
            "source_training_ready", "metadata_valid",
            "preliminary_ready"
        ]
    ].merge(
        lightcurves[
            [
                "tic_key", "fits_key", "file", "sector",
                "status", "n_valid", "fits_found",
                "fits_audit_ok"
            ]
        ],
        on="tic_key",
        how="left",
        validate="many_to_many",
    )

    links["usable_link"] = (
        links["preliminary_ready"]
        & links["fits_audit_ok"]
    )

    DATA.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(OUT_SIGNALS, index=False)
    links.to_csv(OUT_LINKS, index=False)

    summary = (
        manifest.groupby(
            ["signal_label", "preliminary_ready"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 75)
    print("VYSLEDKY")
    print("=" * 75)

    print("Celkem signalu:", len(manifest))
    print("Unikatnich TIC:", manifest["tic_key"].nunique())
    print(
        "Training ready minimum = yes:",
        int(manifest["source_training_ready"].sum())
    )
    print(
        "Platna metadata:",
        int(manifest["metadata_valid"].sum())
    )
    print(
        "Signal s audit-ok FITS:",
        int(manifest["has_audit_ok_fits"].sum())
    )
    print(
        "Predbezne pripravenych signalu:",
        int(manifest["preliminary_ready"].sum())
    )
    print(
        "FITS propojenych s auditem:",
        int(lightcurves["fits_found"].sum())
    )
    print(
        "FITS audit-ok:",
        int(lightcurves["fits_audit_ok"].sum())
    )
    print(
        "Pouzitelnych vazeb signal-FITS:",
        int(links["usable_link"].sum())
    )

    print("\nROZLOZENI TRID:")
    print(
        pd.crosstab(
            manifest["signal_label"],
            manifest["preliminary_ready"],
        ).to_string()
    )

    print("\nVYSTUPY:")
    print(OUT_SIGNALS)
    print(OUT_LINKS)
    print(OUT_SUMMARY)

    print("\nHOTOVO")


if __name__ == "__main__":
    main()
