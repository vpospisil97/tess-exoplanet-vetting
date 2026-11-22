"""Připravuje úplný manifest dostupných signálů a FITS souborů."""


from pathlib import Path
import shutil

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent

SIGNALS_FILE = BASE / "signals.csv"
LIGHTCURVES_FILE = BASE / "lightcurves.csv"

PILOT_DIR = BASE / "TESS_sample_150"
PILOT_MANIFEST = (
    PILOT_DIR / "tess_sample_150_manifest.csv"
)

OUTPUT_DIR = BASE / "full_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

SIGNAL_OUTPUT = OUTPUT_DIR / "eligible_signals.csv"
FITS_OUTPUT = OUTPUT_DIR / "fits_download_manifest.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "dataset_summary.csv"

BASE_URL = (
    "https://huggingface.co/datasets/"
    "saadtaleb/tess-lightcurves-planets/"
    "resolve/main/"
)

LABELS = [
    "candidate",
    "confirmed",
    "false_positive",
]


def normalize_tic(value):
    """Sjednotí identifikátor TIC v katalogových datech."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.startswith("TIC_"):
        text = text[4:]

    if text.endswith(".0"):
        text = text[:-2]

    return text


def main():
    """Vytvoří manifest signálů a adres FITS pro úplný dataset."""
    print("=" * 70)
    print("SESTAVENI KOMPLETNIHO TESS MANIFESTU")
    print("=" * 70)

    signals = pd.read_csv(
        SIGNALS_FILE,
        low_memory=False
    )

    lightcurves = pd.read_csv(
        LIGHTCURVES_FILE,
        low_memory=False
    )

    # Pilot manifest is optional. A fresh clone can generate the full
    # download manifest without any local pilot artifacts.
    if PILOT_MANIFEST.is_file():
        pilot = pd.read_csv(
            PILOT_MANIFEST,
            low_memory=False
        )

        required_pilot = {"tic_id", "hf_path"}
        if not required_pilot.issubset(pilot.columns):
            raise ValueError(
                "Pilotni manifest nema povinne sloupce "
                "tic_id a hf_path."
            )

        pilot["tic_id"] = pilot["tic_id"].map(
            normalize_tic
        )
        pilot_paths = set(pilot["hf_path"].astype(str))
        print("Pilotni manifest nalezen:", PILOT_MANIFEST)
    else:
        pilot_paths = set()
        print(
            "Pilotni manifest nebyl nalezen; "
            "vsechny FITS budou vedeny jako mimo pilot."
        )

    for df in [signals, lightcurves]:
        df["tic_id"] = df["tic_id"].map(
            normalize_tic
        )

    required_signals = {
        "signal_uid",
        "tic_id",
        "signal_label",
        "period_days",
        "epoch_bjd",
        "duration_hours",
        "training_ready_minimum",
    }

    required_lightcurves = {
        "tic_id",
        "hf_path",
        "sector",
        "file_type",
    }

    if not required_signals.issubset(signals.columns):
        raise ValueError(
            "Chybi povinne sloupce v signals.csv."
        )

    if not required_lightcurves.issubset(
        lightcurves.columns
    ):
        raise ValueError(
            "Chybi povinne sloupce v lightcurves.csv."
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

    ready = (
        signals["training_ready_minimum"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
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

    valid_labels = (
        signals["signal_label"].isin(LABELS)
    )

    valid_tic = signals["tic_id"] != ""

    eligible = signals[
        ready
        & valid_parameters
        & valid_labels
        & valid_tic
    ].copy()

    # Vsechny signaly jedne hvezdy zachovame.
    # Pri pozdejsim deleni zustanou ve stejne
    # trenovaci, validacni nebo testovaci skupine.

    if eligible["signal_uid"].duplicated().any():
        raise ValueError(
            "Duplicitni signal_uid v metadatech."
        )

    lightcurves = lightcurves[
        lightcurves["tic_id"] != ""
    ].copy()

    lightcurves["hf_path"] = (
        lightcurves["hf_path"].astype(str)
    )

    # Bezpecnostni kontrola relativnich cest.
    def safe_path(value):
        """Ověří, zda relativní cesta FITS neobsahuje zakázané segmenty."""
        path = Path(value.replace("\\", "/"))

        return (
            not path.is_absolute()
            and ".." not in path.parts
            and path.suffix.lower() == ".fits"
            and len(path.parts) == 2
        )

    lightcurves = lightcurves[
        lightcurves["hf_path"].map(safe_path)
    ].copy()

    # Stejna cesta nesmi ukazovat na ruzne TIC.
    conflicts = (
        lightcurves.groupby("hf_path")["tic_id"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            "Konflikt TIC u stejne FITS cesty."
        )

    lightcurves = lightcurves.drop_duplicates(
        subset=["hf_path"]
    )

    available_tics = set(
        lightcurves["tic_id"]
    )

    eligible = eligible[
        eligible["tic_id"].isin(available_tics)
    ].copy()

    eligible = eligible.sort_values(
        ["tic_id", "signal_uid"]
    ).reset_index(drop=True)

    selected_tics = set(eligible["tic_id"])

    fits_manifest = lightcurves[
        lightcurves["tic_id"].isin(selected_tics)
    ].copy()

    fits_manifest = fits_manifest.sort_values(
        ["tic_id", "sector", "hf_path"]
    ).reset_index(drop=True)

    fits_manifest["download_url"] = (
        BASE_URL
        + fits_manifest["hf_path"]
    )

    # Zjistime, ktere FITS uz byly v pilotnim manifestu.
    # Bez pilotniho manifestu jsou vsechny hodnoty False.
    fits_manifest["in_pilot"] = (
        fits_manifest["hf_path"].isin(
            pilot_paths
        )
    )

    # Pokud budeme stahovat dalsi soubory,
    # ulozime je do full_dataset/fits.
    fits_manifest["local_path"] = (
        "full_dataset/fits/"
        + fits_manifest["hf_path"]
    )

    # Pocet signalu pro jednotlive TIC.
    signal_counts = (
        eligible.groupby("tic_id")
        .size()
        .rename("signals_for_tic")
    )

    fits_manifest = fits_manifest.merge(
        signal_counts,
        left_on="tic_id",
        right_index=True,
        how="left"
    )

    eligible.to_csv(
        SIGNAL_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    fits_manifest.to_csv(
        FITS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    counts = (
        eligible["signal_label"]
        .value_counts()
    )

    summary = pd.DataFrame([
        {
            "metric": "all_signals",
            "value": len(signals),
        },
        {
            "metric": "eligible_signals",
            "value": len(eligible),
        },
        {
            "metric": "unique_tics",
            "value": eligible["tic_id"].nunique(),
        },
        {
            "metric": "unique_fits",
            "value": len(fits_manifest),
        },
        {
            "metric": "pilot_fits_in_manifest",
            "value": int(
                fits_manifest["in_pilot"].sum()
            ),
        },
        {
            "metric": "candidate",
            "value": int(
                counts.get("candidate", 0)
            ),
        },
        {
            "metric": "confirmed",
            "value": int(
                counts.get("confirmed", 0)
            ),
        },
        {
            "metric": "false_positive",
            "value": int(
                counts.get("false_positive", 0)
            ),
        },
    ])

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    disk = shutil.disk_usage(BASE)

    print("\nVYSLEDKY")
    print("-" * 70)

    print("Puvodni signaly:", len(signals))
    print("Vhodne podle metadat:", len(eligible))
    print(
        "Unikatni TIC:",
        eligible["tic_id"].nunique()
    )
    print(
        "Unikatni FITS k ziskani:",
        len(fits_manifest)
    )

    print(
        "FITS v pilotnim manifestu:",
        int(fits_manifest["in_pilot"].sum())
    )

    print("\nRozlozeni trid:")
    print(
        eligible["signal_label"]
        .value_counts()
        .to_string()
    )

    print("\nTypy FITS:")
    print(
        fits_manifest["file_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nVolne misto na disku:",
        round(disk.free / 1024**3, 2),
        "GiB"
    )

    print("\nVystupni soubory:")
    print(SIGNAL_OUTPUT)
    print(FITS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nManifest vytvoren.")
    print("Zatim nebyl stazen zadny soubor.")


if __name__ == "__main__":
    main()
