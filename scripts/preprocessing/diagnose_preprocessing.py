from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

BASE = Path(__file__).resolve().parent
FITS_DIR = BASE / "full_dataset" / "fits"
OUTPUT_DIR = BASE / "full_dataset" / "preprocessing_test" / "diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_fits(path, flux_column):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if not isinstance(
                hdu, (fits.BinTableHDU, fits.TableHDU)
            ):
                continue

            if hdu.data is None:
                continue

            names = hdu.columns.names or []

            if "TIME" not in names or flux_column not in names:
                continue

            time = np.array(
                hdu.data["TIME"],
                dtype=np.float64,
                copy=True,
            )

            flux = np.array(
                hdu.data[flux_column],
                dtype=np.float64,
                copy=True,
            )

            quality = (
                np.array(
                    hdu.data["QUALITY"],
                    dtype=np.int64,
                    copy=True,
                )
                if "QUALITY" in names
                else np.zeros(len(time), dtype=np.int64)
            )

            return time, flux, quality

    raise ValueError("Nenalezena vhodna FITS tabulka.")


def analyze(tic, filename, flux_column):
    path = FITS_DIR / tic / filename

    time, flux, quality = load_fits(path, flux_column)

    n = len(time)

    # Kazdemu mereni ponechame jeho puvodni index.
    original_indices = np.arange(n)

    finite = np.isfinite(time) & np.isfinite(flux)
    valid = finite & (time > 0) & (flux > 0)

    valid_indices = original_indices[valid]
    valid_time = time[valid]
    valid_flux = flux[valid]
    valid_quality = quality[valid]

    # Serazeni a odstraneni duplicit presne jako v
    # puvodnim test_preprocessing.py.
    order = np.argsort(valid_time)

    sorted_time = valid_time[order]
    sorted_flux = valid_flux[order]
    sorted_indices = valid_indices[order]

    unique_time, unique_positions = np.unique(
        sorted_time,
        return_index=True,
    )

    unique_flux = sorted_flux[unique_positions]
    unique_indices = sorted_indices[unique_positions]

    median = np.median(unique_flux)
    mad = np.median(np.abs(unique_flux - median))
    sigma = 1.4826 * mad

    if sigma > 0 and np.isfinite(sigma):
        upper_limit = median + 10 * sigma
        keep_unique = unique_flux <= upper_limit
    else:
        upper_limit = np.nan
        keep_unique = np.ones(
            len(unique_flux),
            dtype=bool,
        )

    kept_indices = unique_indices[keep_unique]
    removed_outlier_indices = unique_indices[~keep_unique]

    # Vsechny body, ktere nebyly zachovany.
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept_indices] = True

    removed_mask = ~kept_mask

    # Pro graf zobrazime pouze body s platnym casem a fluxem.
    # Neplatne zaznamy vykazeme samostatne.
    plot_mask = valid

    plot_kept = plot_mask & kept_mask
    plot_removed = plot_mask & removed_mask

    normalization = np.median(flux[kept_indices])

    normalized_flux = np.full(n, np.nan)

    normalized_flux[valid] = (
        flux[valid] / normalization
    )

    print("\n" + "=" * 75)
    print(tic)
    print("=" * 75)

    print("Puvodni pocet:", n)
    print("Finite dvojice:", int(finite.sum()))
    print("Kladny cas a flux:", int(valid.sum()))
    print("Po odstraneni duplicit:", len(unique_time))
    print("Odstranene vysoke extremy:", len(removed_outlier_indices))
    print("Zachovano celkem:", len(kept_indices))
    print("Odstraneno celkem:", int(removed_mask.sum()))
    print("Robustni sigma:", sigma)
    print("Horni limit:", upper_limit)

    removed_rows = pd.DataFrame({
        "original_index": original_indices[removed_mask],
        "time": time[removed_mask],
        "flux": flux[removed_mask],
        "quality": quality[removed_mask],
    })

    removed_csv = OUTPUT_DIR / f"{tic}_removed_points.csv"
    removed_rows.to_csv(removed_csv, index=False)

    print("\nQUALITY odstranene body:")
    print(
        removed_rows["quality"]
        .value_counts()
        .head(10)
        .to_string()
    )

    # Graf 1: celkovy pohled.
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.scatter(
        time[plot_kept],
        normalized_flux[plot_kept],
        s=4,
        alpha=0.45,
        label="Zachovane",
    )

    ax.scatter(
        time[plot_removed],
        normalized_flux[plot_removed],
        s=18,
        color="red",
        alpha=0.8,
        label="Odstranene - platny cas a flux",
    )

    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("TIME (BTJD)")
    ax.set_ylabel("Normalizovany flux")
    ax.set_title(f"{tic} - zachovane a odstranene body")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    output = OUTPUT_DIR / f"{tic}_removed_vs_kept.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print("\nGraf:", output)
    print("CSV:", removed_csv)

    # Graf 2: detaily poklesu a odstranene body.
    if tic == "TIC_98494121":
        windows = [
            (3136.5, 3139.5),
            (3150.5, 3154.5),
        ]

        fig, axes = plt.subplots(
            2, 1, figsize=(15, 9)
        )

        for ax, (start, end) in zip(axes, windows):
            region = (time >= start) & (time <= end)

            ax.scatter(
                time[region & plot_kept],
                normalized_flux[region & plot_kept],
                s=5,
                alpha=0.5,
                label="Zachovane",
            )

            ax.scatter(
                time[region & plot_removed],
                normalized_flux[region & plot_removed],
                s=20,
                color="red",
                alpha=0.8,
                label="Odstranene",
            )

            ax.set_xlim(start, end)
            ax.set_ylim(0.65, 1.08)
            ax.set_ylabel("Normalizovany flux")
            ax.set_title(f"BTJD {start} az {end}")
            ax.legend()
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("TIME (BTJD)")

        fig.tight_layout()

        detail_output = OUTPUT_DIR / f"{tic}_dip_details.png"
        fig.savefig(detail_output, dpi=160)
        plt.close(fig)

        print("Detail:", detail_output)


def main():
    for tic, filename, flux_column in FILES:
        analyze(tic, filename, flux_column)

    print("\nDIAGNOSTIKA DOKONCENA")


if __name__ == "__main__":
    main()
