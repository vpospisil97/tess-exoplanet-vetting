"""Testuje čištění a validitu předzpracovaných měření."""

from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

BASE = Path(__file__).resolve().parent
FITS_DIR = BASE / "full_dataset" / "fits"
OUTPUT_DIR = BASE / "full_dataset" / "preprocessing_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILES = [
    (
        "no_valid_data",
        "TIC_208030798",
        "hlsp_qlp_tess_ffi_s0039-0000000208030798_tess_v01_llc.fits",
    ),
    (
        "duplicate_times",
        "TIC_379240073",
        "tess2023153011303-s0066-0000000379240073-0260-s_lc.fits",
    ),
    (
        "extreme_flux",
        "TIC_98494121",
        "hlsp_qlp_tess_ffi_s0067-0000000098494121_tess_v01_llc.fits",
    ),
]

FLUX_PRIORITY = [
    "PDCSAP_FLUX",
    "KSPSAP_FLUX",
    "DET_FLUX",
]

def robust_scale(values):
    """Odhadne robustní šum a variabilitu měření."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, 1.4826 * mad


def inspect_and_clean(path):
    """Zkontroluje kvalitu a průběh čištění jednoho FITS souboru."""
    result = {
        "file": str(path.relative_to(BASE)),
        "status": "unknown",
        "flux_column": "",
        "n_original": 0,
        "n_finite": 0,
        "n_after_time": 0,
        "n_after_duplicates": 0,
        "n_after_outliers": 0,
        "n_removed_outliers": 0,
        "median_flux": np.nan,
        "robust_sigma": np.nan,
        "normalized_median": np.nan,
        "normalized_min": np.nan,
        "normalized_max": np.nan,
        "error": "",
    }

    try:
        with fits.open(path, memmap=False) as hdul:
            selected = None

            for hdu in hdul:
                if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    continue

                if hdu.data is None:
                    continue

                names = hdu.columns.names or []

                if "TIME" not in names:
                    continue

                for column in FLUX_PRIORITY:
                    if column not in names:
                        continue

                    time = np.asarray(
                        hdu.data["TIME"], dtype=np.float64
                    )
                    flux = np.asarray(
                        hdu.data[column], dtype=np.float64
                    )

                    if time.ndim != 1 or flux.ndim != 1:
                        continue

                    if len(time) != len(flux):
                        continue

                    finite = np.isfinite(time) & np.isfinite(flux)

                    if np.any(finite):
                        selected = (time, flux, column)
                        break

                if selected is not None:
                    break

            if selected is None:
                result["status"] = "no_valid_flux"
                return result, None

            time, flux, column = selected
            result["flux_column"] = column
            result["n_original"] = len(time)

            finite = np.isfinite(time) & np.isfinite(flux)
            result["n_finite"] = int(np.sum(finite))

            time = time[finite]
            flux = flux[finite]

            # TIME=0 je u zkoumaného SPOC souboru výplňová hodnota.
            # Nulový tok také není vhodný pro normalizaci.
            valid = (time > 0) & (flux > 0)

            time = time[valid]
            flux = flux[valid]

            result["n_after_time"] = len(time)

            if len(time) < 100:
                result["status"] = "too_few_points"
                return result, None

            order = np.argsort(time)
            time = time[order]
            flux = flux[order]

            unique_time, unique_indices = np.unique(
                time, return_index=True
            )

            time = unique_time
            flux = flux[unique_indices]

            result["n_after_duplicates"] = len(time)

            if len(time) < 100:
                result["status"] = "too_few_points"
                return result, None

            median, sigma = robust_scale(flux)

            result["median_flux"] = float(median)
            result["robust_sigma"] = float(sigma)

            if not np.isfinite(median) or median <= 0:
                result["status"] = "invalid_median"
                return result, None

            # Jednostranná kontrola vysokých extrémů.
            # Hluboké poklesy ponecháváme kvůli tranzitům.
            if sigma > 0 and np.isfinite(sigma):
                upper_limit = median + 10 * sigma
                keep = flux <= upper_limit
            else:
                keep = np.ones(len(flux), dtype=bool)

            result["n_removed_outliers"] = int(
                np.sum(~keep)
            )

            time = time[keep]
            flux = flux[keep]

            result["n_after_outliers"] = len(time)

            if len(time) < 100:
                result["status"] = "too_few_points"
                return result, None

            normalization = np.median(flux)

            if not np.isfinite(normalization) or normalization <= 0:
                result["status"] = "invalid_normalization"
                return result, None

            normalized_flux = flux / normalization

            result["normalized_median"] = float(
                np.median(normalized_flux)
            )
            result["normalized_min"] = float(
                np.min(normalized_flux)
            )
            result["normalized_max"] = float(
                np.max(normalized_flux)
            )
            result["status"] = "ok"

            cleaned = pd.DataFrame({
                "time": time,
                "normalized_flux": normalized_flux,
            })

            return result, cleaned

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, None


def main():
    """Vyhodnotí funkčnost pilotního předzpracování na vybraných datech."""
    print("=" * 70)
    print("TEST PREDZPRACOVANI TESS SVETELNYCH KRIVEK")
    print("=" * 70)

    results = []

    for label, tic, filename in TEST_FILES:
        path = FITS_DIR / tic / filename

        print(f"\nTEST: {label}")
        print(f"Hvezda: {tic}")

        if not path.exists():
            print("CHYBA: Soubor neexistuje.")
            continue

        result, cleaned = inspect_and_clean(path)
        result["test_case"] = label
        result["tic"] = tic

        results.append(result)

        print("Status:", result["status"])
        print("Flux:", result["flux_column"])
        print("Puvodni pocet:", result["n_original"])
        print("Konecny pocet:", result["n_after_outliers"])
        print("Odstranene vysoke extremy:", result["n_removed_outliers"])

        if cleaned is not None:
            output = OUTPUT_DIR / f"{tic}_cleaned.csv"
            cleaned.to_csv(output, index=False)
            print("Ulozeno:", output)

        if result["error"]:
            print("Chyba:", result["error"])

    report = pd.DataFrame(results)
    report_path = OUTPUT_DIR / "preprocessing_test_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 70)
    print("SOUHRN")
    print("=" * 70)

    print(
        report[
            [
                "tic",
                "status",
                "flux_column",
                "n_original",
                "n_finite",
                "n_after_time",
                "n_after_duplicates",
                "n_after_outliers",
                "n_removed_outliers",
                "normalized_min",
                "normalized_max",
            ]
        ].to_string(index=False)
    )

    print("\nReport:", report_path)
    print("TEST DOKONCEN")


if __name__ == "__main__":
    main()
