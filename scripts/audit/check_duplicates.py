"""Kontroluje opakování hvězd a souborů v pilotním vzorku."""


from pathlib import Path
import pandas as pd

# Slozka projektu
DATA_FOLDER = Path(r"C:\Users\vposp\Desktop\Deeplearning")

# Nacteni predchozi kontrolni tabulky
REPORT_FILE = DATA_FOLDER / "fits_control_report.csv"

df = pd.read_csv(REPORT_FILE)

print("=" * 60)
print("KONTROLA DUPLICIT HVĚZD")
print("=" * 60)

# TIC ID je ulozeno ve slozce jednotlivych souboru
df["tic_id"] = df["tic_folder"].astype(str)

# Pocet souboru a unikatnich hvezd
total_files = len(df)
unique_stars = df["tic_id"].nunique()

print(f"\nCelkem FITS souboru: {total_files}")
print(f"Unikatnich TIC hvezd: {unique_stars}")
print(f"Rozdil: {total_files - unique_stars}")

# Hvezdy s vice soubory
duplicates = (
    df.groupby("tic_id")
    .size()
    .reset_index(name="number_of_files")
)

duplicates = duplicates[
    duplicates["number_of_files"] > 1
].sort_values("number_of_files", ascending=False)

print(f"\nHvezdy s vice FITS soubory: {len(duplicates)}")

if len(duplicates) > 0:
    print("\nNejčastější duplicity:")
    print(duplicates.head(20).to_string(index=False))
else:
    print("\nKazda hvezda ma pouze jeden FITS soubor.")

# Ulozeni vysledku
output_file = DATA_FOLDER / "fits_duplicate_report.csv"
duplicates.to_csv(output_file, index=False)

print(f"\nReport ulozen: {output_file}")
print("\nKontrola dokoncena.")
