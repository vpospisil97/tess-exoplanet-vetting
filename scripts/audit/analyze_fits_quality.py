"""Zkoumá základní statistiky a kvalitu měření v souborech FITS."""


from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


# ============================================================
# NASTAVENI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "TESS_sample_150"

REPORT_FILE = BASE_DIR / "fits_quality_report.csv"
SUMMARY_FILE = BASE_DIR / "fits_quality_summary.csv"


# ============================================================
# POMOCNE FUNKCE
# ============================================================

def get_numeric_column(data, column_name):
    """Nacte sloupec z FITS jako numericke pole."""
    values = np.ma.asarray(data[column_name])
    values = np.ma.filled(values, np.nan)
    return np.asarray(values, dtype=np.float64)


def count_duplicates(values):
    """Spocita duplicitni hodnoty."""
    if len(values) == 0:
        return 0

    return len(values) - len(np.unique(values))


# ============================================================
# NALEZENI FITS SOUBORU
# ============================================================

print("=" * 70)
print("PODROBNA KONTROLA KVALITY TESS FITS")
print("=" * 70)

if not SAMPLE_DIR.exists():
    raise SystemExit(f"Adresar neexistuje: {SAMPLE_DIR}")

fits_files = [
    file
    for file in SAMPLE_DIR.rglob("*")
    if file.is_file() and file.suffix.lower() == ".fits"
]

fits_files = sorted(fits_files)

print(f"\nNalezeno FITS souboru: {len(fits_files)}")

if not fits_files:
    raise SystemExit("Nebyly nalezeny zadne FITS soubory.")


# ============================================================
# ANALYZA JEDNOTLIVYCH SOUBORU
# ============================================================

results = []
errors = []

for index, fits_file in enumerate(fits_files, start=1):

    print(f"Analyzuji {index}/{len(fits_files)}: {fits_file.name}")

    try:
        with fits.open(fits_file, memmap=False) as hdul:

            # Najdeme HDU obsahujici potrebne sloupce.
            data = None
            column_map = {}

            for hdu in hdul:
                if hdu.data is None or not hasattr(hdu, "columns"):
                    continue

                names = hdu.columns.names or []
                names_upper = {
                    name.upper(): name for name in names
                }

                required = {
                    "TIME",
                    "PDCSAP_FLUX",
                    "QUALITY",
                }

                if required.issubset(names_upper):
                    data = hdu.data
                    column_map = names_upper
                    break

            if data is None:
                raise ValueError(
                    "Nebyla nalezena tabulka s TIME, "
                    "PDCSAP_FLUX a QUALITY."
                )

            time = get_numeric_column(
                data, column_map["TIME"]
            )

            flux = get_numeric_column(
                data, column_map["PDCSAP_FLUX"]
            )

            quality = get_numeric_column(
                data, column_map["QUALITY"]
            )

            if not (len(time) == len(flux) == len(quality)):
                raise ValueError(
                    "Sloupce TIME, PDCSAP_FLUX a QUALITY "
                    "nemaji stejnou delku."
                )

            total = len(time)

            # ------------------------------------------------
            # KONTROLA NEKONECNYCH A CHYBEJICICH HODNOT
            # ------------------------------------------------

            finite_time = np.isfinite(time)
            finite_flux = np.isfinite(flux)
            finite_quality = np.isfinite(quality)

            invalid_time = int((~finite_time).sum())
            invalid_flux = int((~finite_flux).sum())
            invalid_quality = int((~finite_quality).sum())

            # ------------------------------------------------
            # KONTROLA TOKU
            # ------------------------------------------------

            zero_flux = int(
                (finite_flux & (flux == 0)).sum()
            )

            negative_flux = int(
                (finite_flux & (flux < 0)).sum()
            )

            positive_flux = finite_flux & (flux > 0)

            # ------------------------------------------------
            # KONTROLA QUALITY
            # ------------------------------------------------

            quality_zero = finite_quality & (quality == 0)

            # Pouzitelne mereni:
            # konecny cas, konecny tok, QUALITY = 0,
            # kladny tok.
            usable = (
                finite_time
                & finite_flux
                & finite_quality
                & quality_zero
                & positive_flux
            )

            usable_count = int(usable.sum())

            # ------------------------------------------------
            # KONTROLA CASU
            # ------------------------------------------------

            valid_times = time[finite_time]
            usable_times = time[usable]

            duplicate_times = count_duplicates(valid_times)
            duplicate_usable_times = count_duplicates(usable_times)

            if len(valid_times) > 1:
                time_differences = np.diff(valid_times)

                non_increasing = int(
                    (time_differences <= 0).sum()
                )

                positive_differences = time_differences[
                    time_differences > 0
                ]

                if len(positive_differences) > 0:
                    median_cadence = float(
                        np.median(positive_differences)
                    )

                    large_gaps = int(
                        (
                            positive_differences
                            > 1.5 * median_cadence
                        ).sum()
                    )

                    maximum_gap = float(
                        np.max(positive_differences)
                    )
                else:
                    median_cadence = np.nan
                    large_gaps = 0
                    maximum_gap = np.nan

            else:
                non_increasing = 0
                median_cadence = np.nan
                large_gaps = 0
                maximum_gap = np.nan

            # ------------------------------------------------
            # CASOVE A TOKOVE ROZSAHY
            # ------------------------------------------------

            if len(valid_times) > 0:
                time_min = float(np.min(valid_times))
                time_max = float(np.max(valid_times))
            else:
                time_min = np.nan
                time_max = np.nan

            valid_flux = flux[usable]

            if len(valid_flux) > 0:
                flux_median = float(np.median(valid_flux))
                flux_min = float(np.min(valid_flux))
                flux_max = float(np.max(valid_flux))
            else:
                flux_median = np.nan
                flux_min = np.nan
                flux_max = np.nan

            # ------------------------------------------------
            # CELKOVY STAV SOUBORU
            # ------------------------------------------------

            problems = []

            if total == 0:
                problems.append("EMPTY_FILE")

            if usable_count == 0:
                problems.append("NO_USABLE_MEASUREMENTS")

            if duplicate_usable_times > 0:
                problems.append("DUPLICATE_USABLE_TIMES")

            if non_increasing > 0:
                problems.append("TIME_NOT_STRICTLY_INCREASING")

            results.append({
                "tic_id": fits_file.parent.name,
                "file_name": fits_file.name,
                "total_measurements": total,
                "invalid_time": invalid_time,
                "invalid_flux": invalid_flux,
                "invalid_quality": invalid_quality,
                "zero_flux": zero_flux,
                "negative_flux": negative_flux,
                "quality_zero": int(quality_zero.sum()),
                "usable_measurements": usable_count,
                "usable_percentage": (
                    100.0 * usable_count / total
                    if total > 0 else 0.0
                ),
                "duplicate_times": duplicate_times,
                "duplicate_usable_times": duplicate_usable_times,
                "non_increasing_time_steps": non_increasing,
                "median_cadence_days": median_cadence,
                "large_time_gaps": large_gaps,
                "maximum_time_gap_days": maximum_gap,
                "time_min": time_min,
                "time_max": time_max,
                "flux_median": flux_median,
                "flux_min": flux_min,
                "flux_max": flux_max,
                "successful": len(problems) == 0,
                "problems": ";".join(problems),
            })

    except Exception as error:
        errors.append({
            "file_name": fits_file.name,
            "error": str(error),
        })

        print(f"  CHYBA: {error}")


# ============================================================
# VYTVORENI REPORTU
# ============================================================

report = pd.DataFrame(results)
error_report = pd.DataFrame(errors)

if len(report) > 0:
    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

if len(error_report) > 0:
    error_file = BASE_DIR / "fits_quality_errors.csv"

    error_report.to_csv(
        error_file,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# SOUHRN
# ============================================================

print("\n" + "=" * 70)
print("SOUHRN KONTROLY")
print("=" * 70)

if len(report) == 0:
    print("Nebyl uspesne analyzovan zadny FITS soubor.")
    raise SystemExit(1)

print(f"Analyzovane soubory:       {len(report)}")
print(f"Soubory s chybou cteni:    {len(errors)}")
print(
    f"Soubory bez zaznamenanych problemu: "
    f"{int(report['successful'].sum())}"
)

print(
    f"Celkem mereni:             "
    f"{int(report['total_measurements'].sum()):,}"
)

print(
    f"Pouzitelna mereni:         "
    f"{int(report['usable_measurements'].sum()):,}"
)

print(
    f"Neplatne casy:             "
    f"{int(report['invalid_time'].sum()):,}"
)

print(
    f"Neplatne hodnoty toku:     "
    f"{int(report['invalid_flux'].sum()):,}"
)

print(
    f"Nulove hodnoty toku:       "
    f"{int(report['zero_flux'].sum()):,}"
)

print(
    f"Zaporne hodnoty toku:      "
    f"{int(report['negative_flux'].sum()):,}"
)

print(
    f"Duplicitni pouzitelne casy:"
    f" {int(report['duplicate_usable_times'].sum()):,}"
)

print(
    f"Nekladne casove kroky:     "
    f"{int(report['non_increasing_time_steps'].sum()):,}"
)

print(
    f"Velke casove mezery:       "
    f"{int(report['large_time_gaps'].sum()):,}"
)

summary = pd.DataFrame([
    {
        "metric": "Analyzovane FITS soubory",
        "value": len(report),
    },
    {
        "metric": "Soubory s chybou cteni",
        "value": len(errors),
    },
    {
        "metric": "Soubory bez zaznamenanych problemu",
        "value": int(report["successful"].sum()),
    },
    {
        "metric": "Celkem mereni",
        "value": int(report["total_measurements"].sum()),
    },
    {
        "metric": "Pouzitelna mereni",
        "value": int(report["usable_measurements"].sum()),
    },
    {
        "metric": "Neplatne casy",
        "value": int(report["invalid_time"].sum()),
    },
    {
        "metric": "Neplatne hodnoty toku",
        "value": int(report["invalid_flux"].sum()),
    },
    {
        "metric": "Duplicitni pouzitelne casy",
        "value": int(report["duplicate_usable_times"].sum()),
    },
    {
        "metric": "Nekladne casove kroky",
        "value": int(report["non_increasing_time_steps"].sum()),
    },
    {
        "metric": "Velke casove mezery",
        "value": int(report["large_time_gaps"].sum()),
    },
])

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\nVytvorene reporty:")
print(REPORT_FILE)
print(SUMMARY_FILE)

if len(errors) > 0:
    print(BASE_DIR / "fits_quality_errors.csv")

print("\nPodrobna kontrola dokoncena.")
