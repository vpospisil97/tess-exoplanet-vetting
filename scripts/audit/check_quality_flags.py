"""Kontroluje příznaky kvality fotometrických měření."""

from pathlib import Path
import numpy as np
from astropy.io import fits

BASE = Path(__file__).resolve().parent

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

for tic, filename, flux_name in FILES:
    path = BASE / "full_dataset" / "fits" / tic / filename

    with fits.open(path, memmap=False) as hdul:
        table = next(
            hdu.data
            for hdu in hdul
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
            and hdu.data is not None
            and "TIME" in (hdu.columns.names or [])
            and flux_name in (hdu.columns.names or [])
            and "QUALITY" in (hdu.columns.names or [])
        )

        time = np.array(table["TIME"], dtype=np.float64)
        flux = np.array(table[flux_name], dtype=np.float64)
        quality = np.array(table["QUALITY"], dtype=np.int64)

    valid = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (time > 0)
        & (flux > 0)
    )

    good_quality = quality == 0

    print("\n" + "=" * 70)
    print(tic)
    print("=" * 70)

    print("Vsechna mereni:", len(time))
    print("Zakladni filtr:", int(valid.sum()))
    print("QUALITY == 0:", int(good_quality.sum()))
    print(
        "Zakladni filtr + QUALITY == 0:",
        int((valid & good_quality).sum())
    )

    print("\nQUALITY u zakladne platnych mereni:")

    values, counts = np.unique(
        quality[valid],
        return_counts=True
    )

    for value, count in zip(values, counts):
        print(f"  {value}: {count}")

    print("\nQUALITY u velmi hlubokych poklesu:")

    if valid.any():
        median = np.median(flux[valid])
        normalized = flux / median

        deep = valid & (normalized < 0.9)

        print("Pocet poklesu pod 0.9:", int(deep.sum()))
        print(
            "Z nich QUALITY == 0:",
            int((deep & good_quality).sum())
        )

        values, counts = np.unique(
            quality[deep],
            return_counts=True
        )

        for value, count in zip(values, counts):
            print(f"  {value}: {count}")

print("\nKONTROLA DOKONCENA")
