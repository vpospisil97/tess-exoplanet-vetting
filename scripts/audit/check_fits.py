"""Provádí základní čitelnostní kontrolu pilotních souborů FITS."""


from pathlib import Path

import pandas as pd
from astropy.io import fits


# --------------------------------------------------
# 1. Nastaveni cest
# --------------------------------------------------

DATA_FOLDER = Path(r"C:\Users\vposp\Desktop\Deeplearning")
OUTPUT_FILE = DATA_FOLDER / "fits_control_report.csv"


# --------------------------------------------------
# 2. Vyhledani FITS souboru
# --------------------------------------------------

fits_files = sorted(DATA_FOLDER.rglob("*.fits"))

print("=" * 60)
print("KONTROLA TESS FITS DATASETU")
print("=" * 60)

print(f"Nalezeno FITS souboru: {len(fits_files)}")


# --------------------------------------------------
# 3. Kontrola jednotlivych souboru
# --------------------------------------------------

results = []

for index, file_path in enumerate(fits_files, start=1):

    print(f"[{index}/{len(fits_files)}] Kontroluji: {file_path.name}")

    result = {
        "file": file_path.name,
        "tic_folder": file_path.parent.name,
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
        "valid_fits": False,
        "has_time": False,
        "has_pdcsap_flux": False,
        "has_quality": False,
        "total_measurements": 0,
        "usable_measurements": 0,
        "error": ""
    }

    try:
        with fits.open(file_path, memmap=False) as hdul:

            # Prohledame jednotlive FITS extensions.
            for hdu in hdul:

                if hdu.data is None:
                    continue

                column_names = []

                if hasattr(hdu.data, "names") and hdu.data.names:
                    column_names = list(hdu.data.names)

                if not column_names:
                    continue

                result["valid_fits"] = True

                # Zajimaji nas hlavne tabulkova astronomicka data.
                upper_names = {
                    name.upper(): name for name in column_names
                }

                if "TIME" in upper_names:
                    result["has_time"] = True

                if "PDCSAP_FLUX" in upper_names:
                    result["has_pdcsap_flux"] = True

                if "QUALITY" in upper_names:
                    result["has_quality"] = True

                required = {"TIME", "PDCSAP_FLUX"}

                if required.issubset(upper_names.keys()):

                    time = hdu.data[upper_names["TIME"]]
                    flux = hdu.data[upper_names["PDCSAP_FLUX"]]

                    result["total_measurements"] = len(time)

                    usable = (
                        pd.notna(time)
                        & pd.notna(flux)
                    )

                    if "QUALITY" in upper_names:
                        quality = hdu.data[upper_names["QUALITY"]]
                        usable = usable & (quality == 0)

                    usable = usable & (flux > 0)

                    result["usable_measurements"] = int(usable.sum())

                    break

    except Exception as error:
        result["error"] = str(error)

    results.append(result)


# --------------------------------------------------
# 4. Vytvoreni kontrolni tabulky
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 5. Souhrn vysledku
# --------------------------------------------------

print("\n" + "=" * 60)
print("VYSLEDEK KONTROLY")
print("=" * 60)

print(f"Celkem souboru: {len(df)}")
print(f"Platne FITS: {df['valid_fits'].sum()}")
print(f"Obsahuji TIME: {df['has_time'].sum()}")
print(f"Obsahuji PDCSAP_FLUX: {df['has_pdcsap_flux'].sum()}")
print(f"Obsahuji QUALITY: {df['has_quality'].sum()}")

print(
    "Celkem pouzitelnych mereni: "
    f"{df['usable_measurements'].sum():,}"
)

print(
    "Soubory s chybou: "
    f"{(df['error'] != '').sum()}"
)

print(f"\nKontrolni tabulka ulozena zde:\n{OUTPUT_FILE}")

print("\nKontrola dokoncena.")
