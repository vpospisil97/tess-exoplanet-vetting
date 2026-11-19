
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

OUTPUT_DIR = BASE_DIR / "prepared_lightcurves"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_DATA = OUTPUT_DIR / "lightcurves_128.npz"
OUTPUT_REPORT = OUTPUT_DIR / "preparation_report.csv"

N_BINS = 128

# Zobrazime oblast +/- 3 delky tranzitu.
WINDOW_MULTIPLIER = 3.0

MIN_TRANSIT_POINTS = 10
MIN_BASELINE_POINTS = 30

# Ochrana pred extremnimi hodnotami.
# Orezavame az po normalizaci a zaznamename to.
CLIP_LIMIT = 10.0


# ============================================================
# POMOCNE FUNKCE
# ============================================================

def normalize_tic(value):
    text = str(value).strip()

    if text.startswith("TIC_"):
        text = text[4:]

    if text.endswith(".0"):
        text = text[:-2]

    return text


def number(value):
    return pd.to_numeric(value, errors="coerce")


def read_reference(hdul):
    for header in (hdul[1].header, hdul[0].header):
        ref_i = header.get("BJDREFI")
        ref_f = header.get("BJDREFF")

        if ref_i is not None or ref_f is not None:
            return float(ref_i or 0) + float(ref_f or 0)

    return None


def robust_scale(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    return 1.4826 * mad


def prepare_one(row, signals_by_uid):

    tic = normalize_tic(row["tic_id"])
    uid = str(row["signal_uid"])
    label = str(row["signal_label"])

    info = {
        "tic_id": tic,
        "signal_uid": uid,
        "signal_label": label,
        "status": "skipped",
        "reason": "",
        "usable_points": 0,
        "transit_points": 0,
        "baseline_points": 0,
        "filled_bins": 0,
        "observed_cycles": 0,
        "clipped_points": 0,
    }

    # --------------------------------------------------------
    # 1. KONTROLA METADAT
    # --------------------------------------------------------

    if uid not in signals_by_uid.index:
        info["reason"] = "missing_signal"
        return None, None, info

    signal = signals_by_uid.loc[uid]

    if isinstance(signal, pd.DataFrame):
        info["reason"] = "duplicate_signal_uid"
        return None, None, info

    if normalize_tic(signal["tic_id"]) != tic:
        info["reason"] = "tic_mismatch"
        return None, None, info

    if str(signal["signal_label"]) != label:
        info["reason"] = "label_mismatch"
        return None, None, info

    period = number(signal["period_days"])
    epoch = number(signal["epoch_bjd"])
    duration = number(signal["duration_hours"])

    if (
        not np.isfinite(period)
        or not np.isfinite(epoch)
        or not np.isfinite(duration)
        or period <= 0
        or duration <= 0
    ):
        info["reason"] = "invalid_transit_parameters"
        return None, None, info

    if duration >= period * 24:
        info["reason"] = "invalid_duration"
        return None, None, info

    # --------------------------------------------------------
    # 2. NACTENI FITS
    # --------------------------------------------------------

    folder = SAMPLE_DIR / f"TIC_{tic}"
    files = sorted(folder.glob("*.fits"))

    if len(files) != 1:
        info["reason"] = "invalid_fits_count"
        return None, None, info

    try:
        with fits.open(files[0], memmap=False) as hdul:

            reference = read_reference(hdul)

            if reference is None:
                info["reason"] = "missing_time_reference"
                return None, None, info

            data = hdul[1].data

            time = np.asarray(
                data["TIME"], dtype=float
            )

            flux = np.asarray(
                data["PDCSAP_FLUX"], dtype=float
            )

            quality = np.asarray(data["QUALITY"])

    except Exception as error:
        info["reason"] = (
            f"fits_error_{type(error).__name__}"
        )
        return None, None, info

    valid = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (flux > 0)
        & (quality == 0)
    )

    time = time[valid] + reference
    flux = flux[valid]

    info["usable_points"] = len(flux)

    if len(flux) == 0:
        info["reason"] = "no_usable_points"
        return None, None, info

    # --------------------------------------------------------
    # 3. FAZOVE SLOZENI
    # --------------------------------------------------------

    phase_days = (
        (time - epoch + period / 2) % period
    ) - period / 2

    phase_hours = phase_days * 24

    # Prevedeme cas na jednotky delky tranzitu.
    # Hodnota 0 = stred tranzitu.
    phase_normalized = phase_hours / duration

    half_period_normalized = (
        period * 12 / duration
    )

    window = min(
        WINDOW_MULTIPLIER,
        half_period_normalized
    )

    # Tranzit je v intervalu -0.5 az +0.5.
    transit_mask = (
        np.abs(phase_normalized) <= 0.5
    )

    # Referencni oblast mimo tranzit.
    baseline_mask = (
        (np.abs(phase_normalized) >= 1.5)
        & (np.abs(phase_normalized) <= window)
    )

    info["transit_points"] = int(
        transit_mask.sum()
    )

    info["baseline_points"] = int(
        baseline_mask.sum()
    )

    if info["transit_points"] < MIN_TRANSIT_POINTS:
        info["reason"] = "too_few_transit_points"
        return None, None, info

    if info["baseline_points"] < MIN_BASELINE_POINTS:
        info["reason"] = "too_few_baseline_points"
        return None, None, info

    cycles = np.floor(
        (time[transit_mask] - epoch) / period + 0.5
    ).astype(np.int64)

    info["observed_cycles"] = len(
        np.unique(cycles)
    )

    # --------------------------------------------------------
    # 4. NORMALIZACE
    # --------------------------------------------------------

    # Zakladni normalizace relativniho toku.
    median_flux = np.median(flux)

    relative_flux = (
        flux / median_flux - 1.0
    )

    # Baseline a robustni meritko pocitame
    # pouze z okoli mimo ocekavany tranzit.
    baseline_values = relative_flux[baseline_mask]

    baseline = np.median(baseline_values)
    scale = robust_scale(baseline_values)

    if not np.isfinite(scale) or scale <= 0:
        info["reason"] = "invalid_scale"
        return None, None, info

    normalized_flux = (
        relative_flux - baseline
    ) / scale

    # --------------------------------------------------------
    # 5. VYBER OKNA A BINOVANI
    # --------------------------------------------------------

    selected = (
        np.abs(phase_normalized) <= window
    )

    selected_phase = phase_normalized[selected]
    selected_flux = normalized_flux[selected]

    # Spolecna osa -3 az +3.
    # U kratkych period mohou okrajove biny
    # zustat prazdne, coz zachyti maska.
    edges = np.linspace(
        -WINDOW_MULTIPLIER,
        WINDOW_MULTIPLIER,
        N_BINS + 1
    )

    bin_index = np.searchsorted(
        edges,
        selected_phase,
        side="right"
    ) - 1

    # Zahrneme i presny pravy okraj.
    bin_index[bin_index == N_BINS] = N_BINS - 1

    values = np.zeros(
        N_BINS,
        dtype=np.float32
    )

    mask = np.zeros(
        N_BINS,
        dtype=np.float32
    )

    clipped_points = 0

    for i in range(N_BINS):

        points = selected_flux[
            bin_index == i
        ]

        if len(points) == 0:
            continue

        median_value = float(np.median(points))

        if abs(median_value) > CLIP_LIMIT:
            clipped_points += 1

        values[i] = np.clip(
            median_value,
            -CLIP_LIMIT,
            CLIP_LIMIT
        )

        mask[i] = 1.0

    info["filled_bins"] = int(mask.sum())
    info["clipped_points"] = clipped_points

    # Musime mit obsazene alespon nejake biny
    # uvnitr ocekavaneho tranzitu.
    centers = (edges[:-1] + edges[1:]) / 2

    transit_bins = np.abs(centers) <= 0.5

    if mask[transit_bins].sum() == 0:
        info["reason"] = "no_filled_transit_bins"
        return None, None, info

    info["status"] = "prepared"
    info["reason"] = ""

    return values, mask, info


# ============================================================
# HLAVNI PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("PRIPRAVA SVETELNYCH KRIVEK PRO DEEP LEARNING")
    print("=" * 70)

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
    }

    if not required_manifest.issubset(manifest.columns):
        raise ValueError(
            "Manifest nema vsechny potrebne sloupce."
        )

    if not required_signals.issubset(signals.columns):
        raise ValueError(
            "signals.csv nema vsechny potrebne sloupce."
        )

    signals["signal_uid"] = (
        signals["signal_uid"].astype(str)
    )

    signals_by_uid = signals.set_index(
        "signal_uid"
    )

    all_values = []
    all_masks = []
    all_labels = []
    all_tics = []
    all_uids = []
    report_rows = []

    total = len(manifest)

    for index, (_, row) in enumerate(
        manifest.iterrows(), start=1
    ):

        values, mask, info = prepare_one(
            row,
            signals_by_uid
        )

        report_rows.append(info)

        if info["status"] == "prepared":

            all_values.append(values)
            all_masks.append(mask)
            all_labels.append(info["signal_label"])
            all_tics.append(info["tic_id"])
            all_uids.append(info["signal_uid"])

        if index % 25 == 0 or index == total:
            print(
                f"Zpracovano: {index}/{total}"
            )

    report = pd.DataFrame(report_rows)

    report.to_csv(
        OUTPUT_REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    # Ulozime vzdy 2D pole, i kdyby nebyl
    # pripraven zadny signal.
    if all_values:
        X = np.stack(all_values).astype(np.float32)
        M = np.stack(all_masks).astype(np.float32)
    else:
        X = np.empty((0, N_BINS), dtype=np.float32)
        M = np.empty((0, N_BINS), dtype=np.float32)

    y = np.asarray(all_labels, dtype=str)
    tics = np.asarray(all_tics, dtype=str)
    uids = np.asarray(all_uids, dtype=str)

    np.savez_compressed(
        OUTPUT_DATA,
        X=X,
        mask=M,
        y=y,
        tic_id=tics,
        signal_uid=uids,
    )

    prepared = report[
        report["status"] == "prepared"
    ]

    skipped = report[
        report["status"] == "skipped"
    ]

    print("\n" + "=" * 70)
    print("VYSLEDKY PRIPRAVY")
    print("=" * 70)

    print(f"Celkem signalu: {total}")
    print(f"Pripraveno: {len(prepared)}")
    print(f"Nepripraveno: {len(skipped)}")

    print(f"\nTvar vstupu X: {X.shape}")
    print(f"Tvar masky: {M.shape}")
    print(f"Tvar klasifikaci y: {y.shape}")

    print("\nRozlozeni pripravenych trid:")

    if len(prepared):
        print(
            prepared["signal_label"]
            .value_counts()
            .to_string()
        )
    else:
        print("Zadne pripravene signaly.")

    print("\nDuvody nepripravenych signalu:")

    if len(skipped):
        print(
            skipped["reason"]
            .value_counts()
            .to_string()
        )
    else:
        print("Zadne.")

    if len(prepared):
        print(
            "\nMedian obsazenych binu: "
            f"{prepared['filled_bins'].median():.1f}"
            f" / {N_BINS}"
        )

        print(
            "Median pozorovanych cyklu: "
            f"{prepared['observed_cycles'].median():.1f}"
        )

    print("\nVystupni soubory:")
    print(OUTPUT_DATA)
    print(OUTPUT_REPORT)

    print("\nPriprava dokoncena.")


if __name__ == "__main__":
    main()
