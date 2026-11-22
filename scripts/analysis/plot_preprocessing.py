"""Vykresluje diagnostické grafy předzpracování světelných křivek."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

BASE = Path(__file__).resolve().parent
FITS_DIR = BASE / "full_dataset" / "fits"
CLEAN_DIR = BASE / "full_dataset" / "preprocessing_test"
PLOT_DIR = CLEAN_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (
        "TIC_379240073",
        "tess2023153011303-s0066-0000000379240073-0260-s_lc.fits",
        "PDCSAP_FLUX",
    ),
    (
        "TIC_98494121",
        "hlsp_qlp_tess_ffi_s0067-0000000098494121_tess_v01_llc.fits",
        "DET_FLUX",
    ),
]

for tic, filename, flux_column in FILES:
    path = FITS_DIR / tic / filename
    clean_path = CLEAN_DIR / f"{tic}_cleaned.csv"

    with fits.open(path, memmap=False) as hdul:
        table = next(
            hdu.data
            for hdu in hdul
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
            and hdu.data is not None
            and "TIME" in (hdu.columns.names or [])
            and flux_column in (hdu.columns.names or [])
        )

        time = np.asarray(table["TIME"], dtype=float)
        flux = np.asarray(table[flux_column], dtype=float)

    clean = pd.read_csv(clean_path)

    finite = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (time > 0)
        & (flux > 0)
    )

    raw_time = time[finite]
    raw_flux = flux[finite]

    raw_median = np.median(raw_flux)
    raw_normalized = raw_flux / raw_median

    # Pro vykresleni pouzijeme omezenou osu Y.
    # Hodnoty mimo rozsah neodstranujeme ze statistik.
    low, high = np.percentile(raw_normalized, [1, 99])

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True
    )

    axes[0].scatter(
        raw_time,
        raw_normalized,
        s=2,
        alpha=0.5,
    )
    axes[0].set_title(f"{tic} - pred cistenim")
    axes[0].set_ylabel("Normalizovany flux")
    axes[0].set_ylim(low - 0.02, high + 0.02)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        clean["time"],
        clean["normalized_flux"],
        s=2,
        alpha=0.5,
    )
    axes[1].set_title(f"{tic} - po cisteni")
    axes[1].set_xlabel("TIME (BTJD)")
    axes[1].set_ylabel("Normalizovany flux")
    axes[1].set_ylim(low - 0.02, high + 0.02)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    output = PLOT_DIR / f"{tic}_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"\n{tic}")
    print("Puvodnich platnych bodu:", len(raw_time))
    print("Vycistenych bodu:", len(clean))
    print("Graf:", output)

print("\nHOTOVO")
