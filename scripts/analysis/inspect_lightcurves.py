"""Prohlíží pilotní světelné křivky a jejich základní charakteristiky."""


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

REPORT_FILE = BASE_DIR / "fits_quality_report.csv"
OUTPUT_DIR = BASE_DIR / "lightcurve_inspection"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# VYBER REPRESENTATIVNICH SOUBORU
# ============================================================

report = pd.read_csv(REPORT_FILE)

selection = {
    "nejvyssi_maximalni_tok": report.loc[
        report["flux_max"].idxmax()
    ],
    "nejnizsi_median_toku": report.loc[
        report["flux_median"].idxmin()
    ],
    "nejvetsi_casova_mezera": report.loc[
        report["maximum_time_gap_days"].idxmax()
    ],
    "nejmensi_podil_pouzitelnych_mereni": report.loc[
        report["usable_percentage"].idxmin()
    ],
}

# Odstraneni pripadnych duplicit
selected = {}

for reason, row in selection.items():
    tic = str(row["tic_id"]).replace("TIC_", "")

    if tic not in selected:
        selected[tic] = {
            "reason": reason,
            "row": row,
        }


# ============================================================
# ANALYZA A VYKRESLENI
# ============================================================

print("=" * 70)
print("VIZUALNI KONTROLA SVETELNYCH KRIVEK")
print("=" * 70)

for tic, item in selected.items():

    row = item["row"]
    reason = item["reason"]

    fits_files = list(
        (SAMPLE_DIR / f"TIC_{tic}").glob("*.fits")
    )

    if not fits_files:
        print(f"\nCHYBI FITS SOUBOR pro TIC {tic}")
        continue

    fits_path = fits_files[0]

    print(f"\nTIC: {tic}")
    print(f"Duvod vyberu: {reason}")
    print(f"FITS: {fits_path.name}")

    with fits.open(fits_path) as hdul:

        data = hdul[1].data

        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        quality = np.asarray(data["QUALITY"])

    # Platna casova a tokova mereni
    finite = np.isfinite(time) & np.isfinite(flux)

    # Mereni s QUALITY = 0
    usable = (
        finite
        & (quality == 0)
        & (flux > 0)
    )

    time_usable = time[usable]
    flux_usable = flux[usable]

    if len(time_usable) == 0:
        print("Zadna pouzitelna mereni.")
        continue

    # Casove serazeni
    order = np.argsort(time_usable)

    time_usable = time_usable[order]
    flux_usable = flux_usable[order]

    # Relativni tok v ppm
    flux_median = np.median(flux_usable)

    relative_flux_ppm = (
        flux_usable / flux_median - 1
    ) * 1_000_000

    # Zakladni statistiky
    percentile_01 = np.percentile(relative_flux_ppm, 1)
    percentile_99 = np.percentile(relative_flux_ppm, 99)

    print(f"Pouzitelna mereni: {len(time_usable):,}")
    print(f"Median toku: {flux_median:.4f}")
    print(f"Relativni tok 1. percentil: {percentile_01:.2f} ppm")
    print(f"Relativni tok 99. percentil: {percentile_99:.2f} ppm")
    print(
        "Rozsah relativniho toku: "
        f"{relative_flux_ppm.min():.2f} az "
        f"{relative_flux_ppm.max():.2f} ppm"
    )

    # Dva pohledy: kompletni rozsah a detail
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True
    )

    fig.suptitle(
        f"TESS svetelna krivka – TIC {tic}\n"
        f"Vyber: {reason}",
        fontsize=13
    )

    axes[0].scatter(
        time_usable,
        relative_flux_ppm,
        s=2,
        alpha=0.5
    )

    axes[0].set_ylabel("Relativni tok [ppm]")
    axes[0].set_title("Kompletni rozsah")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        time_usable,
        relative_flux_ppm,
        s=2,
        alpha=0.5
    )

    axes[1].set_ylim(
        percentile_01,
        percentile_99
    )

    axes[1].set_xlabel("BJD - 2457000 [d]")
    axes[1].set_ylabel("Relativni tok [ppm]")
    axes[1].set_title("Detail – 1. az 99. percentil")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"TIC_{tic}_inspection.png"

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Graf ulozen: {output_file}")


print("\n" + "=" * 70)
print("VIZUALNI KONTROLA DOKONCENA")
print("=" * 70)

print(f"Vysledne grafy: {OUTPUT_DIR}")
