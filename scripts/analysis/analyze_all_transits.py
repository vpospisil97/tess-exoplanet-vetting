
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


# ============================================================
# NASTAVENI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_DIR = BASE_DIR / "TESS_sample_150"
MANIFEST_FILE = SAMPLE_DIR / "tess_sample_150_manifest.csv"
SIGNALS_FILE = BASE_DIR / "signals.csv"

OUTPUT_DIR = BASE_DIR / "transit_analysis_150"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "transit_analysis_report.csv"
SUMMARY_FILE = OUTPUT_DIR / "transit_analysis_summary.csv"
REVIEW_FILE = OUTPUT_DIR / "transit_analysis_review.csv"

# Minimalni pocty bodu pro zakladni statistickou kontrolu.
# Nejde o definitivni vedecka kriteria.
MIN_IN_TRANSIT = 10
MIN_OUT_TRANSIT = 30

# Referencni oblast mimo tranzit:
# 1.5 az 3 delky tranzitu od stredu na obe strany.
REFERENCE_INNER = 1.5
REFERENCE_OUTER = 3.0


# ============================================================
# POMOCNE FUNKCE
# ============================================================

def normalize_tic(value):
    """Prevede TIC identifikator na jednotny textovy zapis."""
    text = str(value).strip()

    if text.startswith("TIC_"):
        text = text[4:]

    if text.endswith(".0"):
        text = text[:-2]

    return text


def to_number(value):
    """Prevede hodnotu na cislo; pri chybe vrati NaN."""
    return pd.to_numeric(value, errors="coerce")


def read_time_reference(hdul):
    """Nacte casovou referenci BJD z FITS hlavicky."""

    for header in (hdul[1].header, hdul[0].header):
        ref_i = header.get("BJDREFI")
        ref_f = header.get("BJDREFF")

        if ref_i is not None or ref_f is not None:
            return float(ref_i or 0) + float(ref_f or 0)

    return None


def robust_scatter(values):
    """
    Robustni odhad rozptylu pomoci MAD.
    Vysledek je v jednotkach vstupnich hodnot.
    """
    if len(values) < 2:
        return np.nan

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    return 1.4826 * mad


def analyze_signal(manifest_row, signals_by_uid):

    tic = normalize_tic(manifest_row["tic_id"])
    signal_uid = str(manifest_row["signal_uid"])

    result = {
        "tic_id": tic,
        "signal_uid": signal_uid,
        "signal_label": str(manifest_row["signal_label"]),
        "period_days": np.nan,
        "epoch_bjd": np.nan,
        "duration_hours": np.nan,
        "catalog_depth_ppm": np.nan,
        "usable_measurements": 0,
        "in_transit_measurements": 0,
        "out_transit_measurements": 0,
        "observed_transit_cycles": 0,
        "phase_coverage_fraction": np.nan,
        "estimated_depth_ppm": np.nan,
        "out_transit_scatter_ppm": np.nan,
        "depth_to_scatter_ratio": np.nan,
        "time_reference": np.nan,
        "status": "review",
        "reason": "",
    }

    reasons = []

    # --------------------------------------------------------
    # 1. Metadata
    # --------------------------------------------------------

    if signal_uid not in signals_by_uid.index:
        result["reason"] = "signal_uid_not_found"
        return result

    signal = signals_by_uid.loc[signal_uid]

    # Kontrola duplicit signal_uid
    if isinstance(signal, pd.DataFrame):
        result["reason"] = "duplicate_signal_uid"
        return result

    if normalize_tic(signal["tic_id"]) != tic:
        result["reason"] = "tic_mismatch"
        return result

    if str(signal["signal_label"]) != result["signal_label"]:
        result["reason"] = "label_mismatch"
        return result

    period = to_number(signal["period_days"])
    epoch = to_number(signal["epoch_bjd"])
    duration = to_number(signal["duration_hours"])
    catalog_depth = to_number(signal["depth_ppm"])

    result["period_days"] = period
    result["epoch_bjd"] = epoch
    result["duration_hours"] = duration
    result["catalog_depth_ppm"] = catalog_depth

    if not np.isfinite(period) or period <= 0:
        reasons.append("invalid_period")

    if not np.isfinite(epoch):
        reasons.append("invalid_epoch")

    if not np.isfinite(duration) or duration <= 0:
        reasons.append("invalid_duration")

    if reasons:
        result["reason"] = ";".join(reasons)
        return result

    period_hours = period * 24

    if duration >= period_hours:
        result["reason"] = "duration_not_shorter_than_period"
        return result

    # --------------------------------------------------------
    # 2. FITS
    # --------------------------------------------------------

    fits_dir = SAMPLE_DIR / f"TIC_{tic}"
    fits_files = sorted(fits_dir.glob("*.fits"))

    if len(fits_files) != 1:
        result["reason"] = f"fits_count_{len(fits_files)}"
        return result

    try:
        with fits.open(fits_files[0], memmap=False) as hdul:

            reference = read_time_reference(hdul)

            if reference is None:
                result["reason"] = "missing_bjd_reference"
                return result

            result["time_reference"] = reference

            data = hdul[1].data

            time = np.asarray(data["TIME"], dtype=float)
            flux = np.asarray(
                data["PDCSAP_FLUX"], dtype=float
            )
            quality = np.asarray(data["QUALITY"])

    except Exception as error:
        result["reason"] = (
            f"fits_read_error_{type(error).__name__}"
        )
        return result

    valid = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (flux > 0)
        & (quality == 0)
    )

    time = time[valid] + reference
    flux = flux[valid]

    result["usable_measurements"] = len(flux)

    if len(flux) < MIN_IN_TRANSIT + MIN_OUT_TRANSIT:
        result["reason"] = "too_few_usable_measurements"
        return result

    # --------------------------------------------------------
    # 3. Normalizace toku
    # --------------------------------------------------------

    median_flux = np.median(flux)

    if not np.isfinite(median_flux) or median_flux <= 0:
        result["reason"] = "invalid_median_flux"
        return result

    flux_ppm = (flux / median_flux - 1) * 1_000_000

    # --------------------------------------------------------
    # 4. Phase-folding
    # --------------------------------------------------------

    phase_days = (
        (time - epoch + period / 2) % period
    ) - period / 2

    phase_hours = phase_days * 24

    half_duration = duration / 2

    in_transit = (
        np.abs(phase_hours) <= half_duration
    )

    reference_region = (
        (np.abs(phase_hours) >= REFERENCE_INNER * duration)
        & (
            np.abs(phase_hours)
            <= REFERENCE_OUTER * duration
        )
    )

    # Kontrola, zda se referencni oblast nevejde
    # mimo dostupny interval jedne periody.
    if REFERENCE_OUTER * duration >= period_hours / 2:
        reasons.append("reference_window_too_wide")

    in_values = flux_ppm[in_transit]
    out_values = flux_ppm[reference_region]

    result["in_transit_measurements"] = len(in_values)
    result["out_transit_measurements"] = len(out_values)

    # Pocet ruznych obeznych cyklu, ve kterych
    # mame alespon jedno mereni uvnitr tranzitu.
    cycle_numbers = np.floor(
        (time[in_transit] - epoch) / period + 0.5
    ).astype(np.int64)

    result["observed_transit_cycles"] = len(
        np.unique(cycle_numbers)
    )

    # Pokryti tranzitu: rozdelime jeho ocekavanou
    # delku na 10 intervalu a zjistime, kolik
    # intervalu obsahuje alespon jedno mereni.
    transit_bins = np.linspace(
        -half_duration,
        half_duration,
        11
    )

    occupied_bins = np.histogram(
        phase_hours[in_transit],
        bins=transit_bins
    )[0]

    result["phase_coverage_fraction"] = (
        np.count_nonzero(occupied_bins) / 10
    )

    if len(in_values) < MIN_IN_TRANSIT:
        reasons.append("too_few_in_transit_points")

    if len(out_values) < MIN_OUT_TRANSIT:
        reasons.append("too_few_reference_points")

    if result["observed_transit_cycles"] < 2:
        reasons.append("fewer_than_two_observed_cycles")

    if result["phase_coverage_fraction"] < 0.8:
        reasons.append("low_phase_coverage")

    # --------------------------------------------------------
    # 5. Odhad hloubky a rozptylu
    # --------------------------------------------------------

    if len(in_values) > 0 and len(out_values) > 0:

        estimated_depth = (
            np.median(out_values)
            - np.median(in_values)
        )

        result["estimated_depth_ppm"] = estimated_depth

        scatter = robust_scatter(out_values)

        result["out_transit_scatter_ppm"] = scatter

        if np.isfinite(scatter) and scatter > 0:
            result["depth_to_scatter_ratio"] = (
                estimated_depth / scatter
            )

        if estimated_depth <= 0:
            reasons.append("non_positive_estimated_depth")

    if reasons:
        result["status"] = "review"
        result["reason"] = ";".join(reasons)
    else:
        result["status"] = "basic_checks_passed"
        result["reason"] = ""

    return result


# ============================================================
# HLAVNI PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("AUTOMATICKA ANALYZA 150 TRANZITNICH SIGNALU TESS")
    print("=" * 70)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(SIGNALS_FILE)

    manifest = pd.read_csv(
        MANIFEST_FILE,
        low_memory=False
    )

    signals = pd.read_csv(
        SIGNALS_FILE,
        low_memory=False
    )

    required_manifest = {
        "tic_id",
        "signal_uid",
        "signal_label",
    }

    required_signals = {
        "signal_uid",
        "tic_id",
        "signal_label",
        "period_days",
        "epoch_bjd",
        "duration_hours",
        "depth_ppm",
    }

    if not required_manifest.issubset(manifest.columns):
        raise ValueError(
            "V manifestu chybi povinne sloupce."
        )

    if not required_signals.issubset(signals.columns):
        raise ValueError(
            "V signals.csv chybi povinne sloupce."
        )

    signals["signal_uid"] = (
        signals["signal_uid"].astype(str)
    )

    signals_by_uid = signals.set_index("signal_uid")

    results = []

    total = len(manifest)

    print(f"Zaznamu v manifestu: {total}")
    print(f"Zaznamu v signals.csv: {len(signals)}")
    print()

    for index, (_, row) in enumerate(
        manifest.iterrows(), start=1
    ):

        result = analyze_signal(
            row,
            signals_by_uid
        )

        results.append(result)

        if index % 25 == 0 or index == total:
            print(
                f"Zpracovano: {index}/{total}"
            )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    review = report[
        report["status"] == "review"
    ].copy()

    review.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary_rows = [
        {
            "metric": "total_signals",
            "value": len(report)
        },
        {
            "metric": "basic_checks_passed",
            "value": int(
                (report["status"] == "basic_checks_passed").sum()
            )
        },
        {
            "metric": "review_required",
            "value": len(review)
        },
        {
            "metric": "median_usable_measurements",
            "value": report[
                "usable_measurements"
            ].median()
        },
        {
            "metric": "median_in_transit_measurements",
            "value": report[
                "in_transit_measurements"
            ].median()
        },
        {
            "metric": "median_observed_transit_cycles",
            "value": report[
                "observed_transit_cycles"
            ].median()
        },
        {
            "metric": "median_phase_coverage",
            "value": report[
                "phase_coverage_fraction"
            ].median()
        },
    ]

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("VYSLEDKY")
    print("=" * 70)

    print(f"Celkem signalu: {len(report)}")
    print(
        "Zakladni kontroly splnilo: "
        f"{(report['status'] == 'basic_checks_passed').sum()}"
    )
    print(
        f"K dalsi kontrole: {len(review)}"
    )

    print("\nRozlozeni podle trid:")
    print(
        pd.crosstab(
            report["signal_label"],
            report["status"]
        ).to_string()
    )

    if not review.empty:
        print("\nNejcastejsi duvody kontroly:")

        reasons = (
            review["reason"]
            .str.split(";")
            .explode()
            .value_counts()
        )

        print(reasons.to_string())

    print("\nVystupni soubory:")
    print(REPORT_FILE)
    print(SUMMARY_FILE)
    print(REVIEW_FILE)

    print("\nAnalyza dokoncena.")


if __name__ == "__main__":
    main()
