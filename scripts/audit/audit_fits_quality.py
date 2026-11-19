
from pathlib import Path
import argparse
import os
import time

import numpy as np
import pandas as pd
from astropy.io import fits


BASE = Path(__file__).resolve().parent

PILOT_DIR = BASE / "TESS_sample_150"
FULL_DIR = BASE / "full_dataset" / "fits"

OUTPUT_DIR = BASE / "full_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLUX_COLUMNS = [
    "PDCSAP_FLUX",
    "KSPSAP_FLUX",
    "DET_FLUX",
]

SAVE_EVERY = 100

REPORT_COLUMNS = [
    "file",
    "status",
    "flux_column",
    "n_total",
    "n_valid",
    "valid_fraction",
    "time_span_days",
    "duplicate_times",
    "flux_mean",
    "flux_std",
    "flux_min",
    "flux_max",
    "error",
]


def inspect_fits(path):
    result = {
        "file": str(path.relative_to(BASE)),
        "status": "unknown",
        "flux_column": "",
        "n_total": 0,
        "n_valid": 0,
        "valid_fraction": np.nan,
        "time_span_days": np.nan,
        "duplicate_times": 0,
        "flux_mean": np.nan,
        "flux_std": np.nan,
        "flux_min": np.nan,
        "flux_max": np.nan,
        "error": "",
    }

    try:
        with fits.open(
            path,
            memmap=False,
            lazy_load_hdus=False
        ) as hdul:

            table = None

            for hdu in hdul:
                if not isinstance(
                    hdu,
                    (fits.BinTableHDU, fits.TableHDU)
                ):
                    continue

                names = set(hdu.columns.names or [])

                if "TIME" not in names:
                    continue

                flux_column = next(
                    (
                        col for col in FLUX_COLUMNS
                        if col in names
                    ),
                    None
                )

                if flux_column is None:
                    continue

                table = hdu.data

                if table is None:
                    continue

                time_values = np.asarray(
                    table["TIME"],
                    dtype=np.float64
                )

                flux = np.asarray(
                    table[flux_column],
                    dtype=np.float64
                )

                result["flux_column"] = flux_column
                break

            if table is None:
                result["status"] = "missing_columns"
                return result

            if time_values.ndim != 1 or flux.ndim != 1:
                result["status"] = "invalid_shape"
                return result

            if len(time_values) != len(flux):
                result["status"] = "length_mismatch"
                return result

            result["n_total"] = len(time_values)

            valid = (
                np.isfinite(time_values)
                & np.isfinite(flux)
            )

            valid_time = time_values[valid]
            valid_flux = flux[valid]

            result["n_valid"] = len(valid_time)

            if len(time_values):
                result["valid_fraction"] = (
                    len(valid_time) / len(time_values)
                )

            if len(valid_time) == 0:
                result["status"] = "no_valid_data"
                return result

            result["time_span_days"] = float(
                np.max(valid_time)
                - np.min(valid_time)
            )

            result["duplicate_times"] = int(
                len(valid_time)
                - len(np.unique(valid_time))
            )

            result["flux_mean"] = float(
                np.mean(valid_flux)
            )

            result["flux_std"] = float(
                np.std(valid_flux)
            )

            result["flux_min"] = float(
                np.min(valid_flux)
            )

            result["flux_max"] = float(
                np.max(valid_flux)
            )

            if len(valid_time) < 100:
                result["status"] = "too_few_points"
            elif result["flux_std"] == 0:
                result["status"] = "constant_flux"
            else:
                result["status"] = "ok"

    except Exception as exc:
        result["status"] = "read_error"
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )[:500]

    return result


def save_report(results, report_path):
    """
    Ulozi report nejprve do docasneho souboru.
    Teprve po dokonceni zapisu nahradi puvodni CSV.
    """

    report = pd.DataFrame(
        results,
        columns=REPORT_COLUMNS
    )

    temporary_path = report_path.with_name(
        report_path.name + ".tmp"
    )

    try:
        report.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig"
        )

        os.replace(
            temporary_path,
            report_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return report


def load_existing_report(report_path, available_files):
    """
    Nacte predchozi vysledky a ponecha pouze zaznamy
    odpovidajici aktualne nalezenym FITS souborum.
    """

    if not report_path.exists():
        return []

    print("\nNalezen existujici report.")
    print("Nacitam predchozi vysledky...")

    report = pd.read_csv(
        report_path,
        encoding="utf-8-sig",
        low_memory=False
    )

    missing = [
        column
        for column in REPORT_COLUMNS
        if column not in report.columns
    ]

    if missing:
        raise ValueError(
            "Existujici report nema ocekavane sloupce: "
            + ", ".join(missing)
        )

    if report["file"].isna().any():
        raise ValueError(
            "Existujici report obsahuje prazdne cesty."
        )

    if report["file"].duplicated().any():
        raise ValueError(
            "Existujici report obsahuje duplicitni cesty."
        )

    report = report[
        report["file"].isin(available_files)
    ].copy()

    print(
        "Nacteno predchozich vysledku:",
        len(report)
    )

    return report[REPORT_COLUMNS].to_dict(
        orient="records"
    )


def format_duration(seconds):
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def print_progress(
    completed,
    total,
    processed_this_run,
    started
):
    elapsed = time.monotonic() - started

    percentage = (
        completed / total * 100
        if total
        else 0
    )

    remaining = total - completed

    if processed_this_run > 0:
        seconds_per_file = (
            elapsed / processed_this_run
        )

        eta = format_duration(
            remaining * seconds_per_file
        )
    else:
        eta = "neznamy"

    print(
        f"Zkontrolovano: {completed}/{total} "
        f"({percentage:.2f} %) | "
        f"Tento beh: {processed_this_run} | "
        f"Cas: {format_duration(elapsed)} | "
        f"Zbyva priblizne: {eta}",
        flush=True
    )


def print_summary(report):
    print("\n========== STAVY ==========")

    print(
        report["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\n========== FOTOMETRICKE SLOUPCE =========="
    )

    print(
        report["flux_column"]
        .fillna("")
        .value_counts()
        .to_string()
    )

    print("\n========== SOUHRN ==========")

    numeric_columns = [
        "n_total",
        "n_valid",
        "valid_fraction",
        "time_span_days",
        "duplicate_times",
        "flux_std",
    ]

    numeric_report = report[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print(
        numeric_report.describe().to_string()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot"
    )

    args = parser.parse_args()

    root = (
        PILOT_DIR
        if args.mode == "pilot"
        else FULL_DIR
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Slozka neexistuje: {root}"
        )

    files = sorted(
        root.rglob("*.fits")
    )

    if not files:
        raise RuntimeError(
            "Nebyly nalezeny zadne FITS soubory."
        )

    report_path = (
        OUTPUT_DIR
        / f"fits_quality_{args.mode}.csv"
    )

    available_files = {
        str(path.relative_to(BASE))
        for path in files
    }

    print("=" * 70)
    print("KONTROLA KVALITY TESS FITS")
    print("=" * 70)

    print("Rezim:", args.mode)
    print("Slozka:", root)
    print("Nalezeno FITS:", len(files))
    print("Ukladani po:", SAVE_EVERY, "souborech")
    print("Report:", report_path)

    results = load_existing_report(
        report_path,
        available_files
    )

    completed_files = {
        str(row["file"])
        for row in results
    }

    remaining_files = [
        path
        for path in files
        if str(path.relative_to(BASE))
        not in completed_files
    ]

    print("\nJiz zkontrolovano:", len(results))
    print("Zbyva zkontrolovat:", len(remaining_files))

    if not remaining_files:
        print("\nVsechny soubory jiz maji zaznam.")

        report = pd.DataFrame(
            results,
            columns=REPORT_COLUMNS
        )

        print_summary(report)
        print("\nKONTROLA DOKONCENA.")
        return

    started = time.monotonic()
    processed_this_run = 0
    unsaved = 0

    try:
        for path in remaining_files:
            result = inspect_fits(path)

            results.append(result)

            processed_this_run += 1
            unsaved += 1

            completed = len(results)

            if (
                unsaved >= SAVE_EVERY
                or completed == len(files)
            ):
                save_report(
                    results,
                    report_path
                )

                unsaved = 0

                print_progress(
                    completed,
                    len(files),
                    processed_this_run,
                    started
                )

                print(
                    "Prubezne vysledky ulozeny.",
                    flush=True
                )

    except KeyboardInterrupt:
        print(
            "\n\nKontrola prerusena uzivatelem."
        )

        print("Ukladam dosavadni vysledky...")

        save_report(
            results,
            report_path
        )

        print(
            "Ulozeno zaznamu:",
            len(results)
        )

        print(
            "Pri dalsim spusteni bude kontrola "
            "pokracovat."
        )

        return

    report = save_report(
        results,
        report_path
    )

    print_summary(report)

    print("\nReport:", report_path)

    print(
        "\nCelkovy cas tohoto behu:",
        format_duration(
            time.monotonic() - started
        )
    )

    print("\nKONTROLA DOKONCENA.")


if __name__ == "__main__":
    main()
