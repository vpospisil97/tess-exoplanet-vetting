
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# NASTAVENI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "TESS_sample_150"

SIGNALS_FILE = BASE_DIR / "signals.csv"
MANIFEST_FILE = SAMPLE_DIR / "tess_sample_150_manifest.csv"

OUTPUT_DIR = BASE_DIR / "transit_inspection"
OUTPUT_DIR.mkdir(exist_ok=True)

# Hvezdy, ktere jsme kontrolovali v predchozim kroku
SELECTED_TICS = [
    188768068,
    77228304,
    319068154,
    39926974,
]


# ============================================================
# POMOCNE FUNKCE
# ============================================================

def to_number(value):
    """Prevede hodnotu na cislo, nebo vrati NaN."""
    return pd.to_numeric(value, errors="coerce")


def get_fits_time(time, header):
    """
    Prevede cas FITS na absolutni BJD.

    TESS TIME je obvykle BTJD = BJD - 2457000.
    Referencni hodnota se prednostne cte z FITS hlavicky.
    """

    ref_i = header.get("BJDREFI")
    ref_f = header.get("BJDREFF")

    if ref_i is not None or ref_f is not None:
        reference = float(ref_i or 0) + float(ref_f or 0)
        return time + reference, reference

    # Nektere FITS soubory mohou mit referenci v primarni hlavicce.
    return None, None


def phase_fold(time_bjd, flux_ppm, period, epoch):
    """
    Vypocita casovou vzdalenost od stredu tranzitu.

    Vysledek je v hodinach, v intervalu
    -period/2 az +period/2.
    """

    phase = (
        (time_bjd - epoch + period / 2) % period
    ) - period / 2

    return phase * 24, flux_ppm


# ============================================================
# NACTENI METADAT
# ============================================================

print("=" * 70)
print("KONTROLA TRANZITNICH SIGNALU TESS")
print("=" * 70)

if not SIGNALS_FILE.exists():
    raise FileNotFoundError(f"Chybi {SIGNALS_FILE}")

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(f"Chybi {MANIFEST_FILE}")

signals = pd.read_csv(SIGNALS_FILE, low_memory=False)
manifest = pd.read_csv(MANIFEST_FILE, low_memory=False)

signals["tic_id_numeric"] = pd.to_numeric(
    signals["tic_id"].astype(str).str.replace(
        "TIC_", "", regex=False
    ),
    errors="coerce"
)

manifest["tic_id_numeric"] = pd.to_numeric(
    manifest["tic_id"].astype(str).str.replace(
        "TIC_", "", regex=False
    ),
    errors="coerce"
)

results = []


# ============================================================
# ANALYZA JEDNOTLIVYCH HVEZD
# ============================================================

for tic in SELECTED_TICS:

    print("\n" + "-" * 70)
    print(f"TIC {tic}")
    print("-" * 70)

    manifest_rows = manifest[
        manifest["tic_id_numeric"] == tic
    ]

    if manifest_rows.empty:
        print("Hvezda neni v manifestu.")
        continue

    manifest_row = manifest_rows.iloc[0]

    signal_uid = str(manifest_row["signal_uid"])

    signal_rows = signals[
        signals["signal_uid"].astype(str) == signal_uid
    ]

    if signal_rows.empty:
        print(f"Signal {signal_uid} nebyl nalezen.")
        continue

    signal = signal_rows.iloc[0]

    label = str(signal["signal_label"])
    period = to_number(signal["period_days"])
    epoch = to_number(signal["epoch_bjd"])
    duration = to_number(signal["duration_hours"])
    depth = to_number(signal["depth_ppm"])

    print(f"Signal UID: {signal_uid}")
    print(f"Klasifikace: {label}")
    print(f"Perioda [dny]: {period}")
    print(f"Epocha [BJD]: {epoch}")
    print(f"Delka tranzitu [h]: {duration}")
    print(f"Hloubka [ppm]: {depth}")

    fits_files = list(
        (SAMPLE_DIR / f"TIC_{tic}").glob("*.fits")
    )

    if len(fits_files) != 1:
        print(
            f"Ocekavan jeden FITS soubor, nalezeno: "
            f"{len(fits_files)}"
        )
        continue

    with fits.open(fits_files[0]) as hdul:

        primary_header = hdul[0].header
        data_header = hdul[1].header
        data = hdul[1].data

        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        quality = np.asarray(data["QUALITY"])

        reference = None

        for header in (data_header, primary_header):
            time_result, reference = get_fits_time(
                time, header
            )

            if time_result is not None:
                time_bjd = time_result
                break

        if reference is None:
            print(
                "CHYBA: Ve FITS chybi BJDREFI/BJDREFF. "
                "Casovou soustavu nelze overit."
            )
            continue

    print(f"Casova reference FITS: {reference}")

    valid = (
        np.isfinite(time_bjd)
        & np.isfinite(flux)
        & (flux > 0)
        & (quality == 0)
    )

    time_bjd = time_bjd[valid]
    flux = flux[valid]

    if len(flux) == 0:
        print("Zadna pouzitelna mereni.")
        continue

    median_flux = np.median(flux)

    flux_ppm = (flux / median_flux - 1) * 1_000_000

    if (
        not np.isfinite(period)
        or not np.isfinite(epoch)
        or period <= 0
    ):
        print("Neplatna perioda nebo epocha.")
        continue

    phase_hours, folded_flux = phase_fold(
        time_bjd,
        flux_ppm,
        period,
        epoch
    )

    # Zobrazime okoli stredu tranzitu.
    # Pokud je znama delka, pouzijeme +/- 3 delky.
    # Jinak +/- 10 % cele periody.
    if np.isfinite(duration) and duration > 0:
        window_hours = 3 * duration
    else:
        window_hours = period * 24 * 0.1

    # Okno nesmi presahnout polovinu periody.
    window_hours = min(window_hours, period * 12)

    in_window = np.abs(phase_hours) <= window_hours

    phase_window = phase_hours[in_window]
    flux_window = folded_flux[in_window]

    print(f"Pouzitelna mereni: {len(flux):,}")
    print(f"Mereni v okoli tranzitu: {len(flux_window):,}")

    if len(flux_window) == 0:
        print("V okoli tranzitu nejsou zadna mereni.")
        continue

    # Binovani: median toku v malych intervalech faze.
    bins = np.linspace(
        -window_hours,
        window_hours,
        81
    )

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(phase_window, bins)

    binned_flux = np.full(
        len(bin_centers),
        np.nan
    )

    for i in range(1, len(bins)):
        values = flux_window[bin_indices == i]

        if len(values) > 0:
            binned_flux[i - 1] = np.median(values)

    # ========================================================
    # GRAF
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8)
    )

    fig.suptitle(
        f"TIC {tic} | {label} | perioda {period:.5f} d",
        fontsize=13
    )

    axes[0].scatter(
        time_bjd - 2457000,
        flux_ppm,
        s=2,
        alpha=0.5
    )

    axes[0].set_title("Puvodni svetelna krivka")
    axes[0].set_xlabel("BJD - 2457000 [d]")
    axes[0].set_ylabel("Relativni tok [ppm]")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        phase_window,
        flux_window,
        s=3,
        alpha=0.15,
        label="Jednotliva mereni"
    )

    axes[1].plot(
        bin_centers,
        binned_flux,
        color="red",
        linewidth=2,
        label="Median po binech"
    )

    axes[1].axvline(
        0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Predpokladany stred tranzitu"
    )

    if np.isfinite(duration) and duration > 0:
        axes[1].axvspan(
            -duration / 2,
            duration / 2,
            color="orange",
            alpha=0.15,
            label="Ocekavana delka tranzitu"
        )

    axes[1].set_title("Phase-folding: okoli tranzitu")
    axes[1].set_xlabel("Cas od stredu tranzitu [hodiny]")
    axes[1].set_ylabel("Relativni tok [ppm]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR / f"TIC_{tic}_transit.png"
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Graf ulozen: {output_file}")

    results.append({
        "tic_id": tic,
        "signal_uid": signal_uid,
        "signal_label": label,
        "period_days": period,
        "epoch_bjd": epoch,
        "duration_hours": duration,
        "depth_ppm": depth,
        "fits_time_reference": reference,
        "usable_measurements": len(flux),
        "measurements_near_transit": len(flux_window),
        "graph_file": output_file.name,
    })


# ============================================================
# ULOZENI SOUHRNU
# ============================================================

summary_file = OUTPUT_DIR / "transit_inspection_summary.csv"

pd.DataFrame(results).to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("KONTROLA DOKONCENA")
print("=" * 70)

print(f"Zpracovano signalu: {len(results)}")
print(f"Grafy: {OUTPUT_DIR}")
print(f"Souhrn: {summary_file}")
