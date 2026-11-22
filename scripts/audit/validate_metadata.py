"""Ověřuje návaznost katalogových metadat a lokálních FITS souborů."""

from pathlib import Path
import pandas as pd


# ============================================================
# NASTAVENI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "TESS_sample_150"

MANIFEST_FILE = SAMPLE_DIR / "tess_sample_150_manifest.csv"
SIGNALS_FILE = BASE_DIR / "signals.csv"
LIGHTCURVES_FILE = BASE_DIR / "lightcurves.csv"
TARGETS_FILE = BASE_DIR / "targets.csv"

REPORT_FILE = BASE_DIR / "metadata_validation_report.csv"
ERRORS_FILE = BASE_DIR / "metadata_validation_errors.csv"
SUMMARY_FILE = BASE_DIR / "metadata_validation_summary.csv"


# ============================================================
# POMOCNE FUNKCE
# ============================================================

def normalize_text(value):
    """Prevede hodnotu na ocisteny text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_tic(value):
    """Sjednoti zapis TIC identifikatoru."""
    value = normalize_text(value)

    if value.endswith(".0"):
        value = value[:-2]

    return value


def normalize_sector(value):
    """Sjednoti zapis cisla sektoru."""
    value = normalize_text(value)

    if value.endswith(".0"):
        value = value[:-2]

    return value


def normalize_label(value):
    """Sjednoti zapis klasifikacniho stitku."""
    return normalize_text(value).lower()


def split_semicolon_list(value):
    """Rozdeli seznam hodnot oddelenych strednikem."""
    value = normalize_text(value)

    if not value:
        return []

    return [item.strip() for item in value.split(";") if item.strip()]


def get_filename(path_value):
    """Ziska nazev souboru z cesty ulozene v CSV."""
    path_value = normalize_text(path_value)
    return path_value.replace("\\", "/").split("/")[-1]


def add_problem(problems, condition, message):
    """Prida chybu, pokud kontrolovana podminka neplati."""
    if not condition:
        problems.append(message)


# ============================================================
# KONTROLA VSTUPNICH SOUBORU
# ============================================================

print("=" * 70)
print("KONTROLA NAVAZNOSTI TESS METADAT")
print("=" * 70)

required_files = [
    MANIFEST_FILE,
    SIGNALS_FILE,
    LIGHTCURVES_FILE,
    TARGETS_FILE,
]

missing_files = [file for file in required_files if not file.exists()]

if missing_files:
    print("\nCHYBA: Chybi nasledujici vstupni soubory:")

    for file in missing_files:
        print(f"  - {file}")

    raise SystemExit(1)

if not SAMPLE_DIR.exists():
    print(f"\nCHYBA: Adresar se vzorkem FITS souboru neexistuje: {SAMPLE_DIR}")
    raise SystemExit(1)

print("\nVsechny vstupni CSV soubory nalezeny.")


# ============================================================
# NACITANI METADAT
# ============================================================

print("\nNacitam metadata...")

manifest = pd.read_csv(MANIFEST_FILE, dtype=str, keep_default_na=False)
signals = pd.read_csv(SIGNALS_FILE, dtype=str, keep_default_na=False)
lightcurves = pd.read_csv(LIGHTCURVES_FILE, dtype=str, keep_default_na=False)
targets = pd.read_csv(TARGETS_FILE, dtype=str, keep_default_na=False)

print(f"Manifest:     {len(manifest):,} zaznamu")
print(f"Signals:      {len(signals):,} zaznamu")
print(f"Lightcurves:  {len(lightcurves):,} zaznamu")
print(f"Targets:      {len(targets):,} zaznamu")


# ============================================================
# KONTROLA POVINNYCH SLOUPCU
# ============================================================

required_columns = {
    "manifest": {
        "signal_label",
        "signal_uid",
        "tic_id",
        "sector",
        "hf_path",
    },
    "signals": {
        "signal_uid",
        "tic_id",
        "signal_label",
    },
    "lightcurves": {
        "tic_id",
        "sector",
        "file_name",
        "signal_uids",
    },
    "targets": {
        "tic_id",
    },
}

dataframes = {
    "manifest": manifest,
    "signals": signals,
    "lightcurves": lightcurves,
    "targets": targets,
}

for dataframe_name, columns in required_columns.items():
    missing_columns = columns - set(dataframes[dataframe_name].columns)

    if missing_columns:
        print(
            f"\nCHYBA: V souboru {dataframe_name} chybi sloupce: "
            f"{', '.join(sorted(missing_columns))}"
        )
        raise SystemExit(1)


# ============================================================
# NORMALIZACE IDENTIFIKATORU
# ============================================================

manifest["tic_norm"] = manifest["tic_id"].apply(normalize_tic)
manifest["uid_norm"] = manifest["signal_uid"].apply(normalize_text)
manifest["label_norm"] = manifest["signal_label"].apply(normalize_label)
manifest["sector_norm"] = manifest["sector"].apply(normalize_sector)
manifest["filename_norm"] = manifest["hf_path"].apply(get_filename)

signals["tic_norm"] = signals["tic_id"].apply(normalize_tic)
signals["uid_norm"] = signals["signal_uid"].apply(normalize_text)
signals["label_norm"] = signals["signal_label"].apply(normalize_label)

lightcurves["tic_norm"] = lightcurves["tic_id"].apply(normalize_tic)
lightcurves["sector_norm"] = lightcurves["sector"].apply(normalize_sector)
lightcurves["filename_norm"] = lightcurves["file_name"].apply(
    lambda value: get_filename(value)
)

targets["tic_norm"] = targets["tic_id"].apply(normalize_tic)


# ============================================================
# INDEXY METADAT
# ============================================================

signals_by_uid = {}

for _, row in signals.iterrows():
    uid = row["uid_norm"]

    if uid:
        signals_by_uid.setdefault(uid, []).append(row)


targets_by_tic = set(targets["tic_norm"])

lightcurves_by_key = {}

for _, row in lightcurves.iterrows():
    key = (
        row["tic_norm"],
        row["sector_norm"],
        row["filename_norm"],
    )

    lightcurves_by_key.setdefault(key, []).append(row)


# ============================================================
# INDEX LOKALNICH FITS SOUBORU
# ============================================================

fits_files = [
    file
    for file in SAMPLE_DIR.rglob("*")
    if file.is_file() and file.suffix.lower() == ".fits"
]

local_fits_by_key = {}

for fits_file in fits_files:
    tic_folder = fits_file.parent.name

    if tic_folder.startswith("TIC_"):
        tic = normalize_tic(tic_folder[4:])
    else:
        tic = ""

    filename = fits_file.name

    key = (tic, filename.lower())
    local_fits_by_key.setdefault(key, []).append(fits_file)

print(f"\nLokalni FITS soubory: {len(fits_files)}")
print(f"Zaznamy v manifestu:  {len(manifest)}")


# ============================================================
# KONTROLA JEDNOTLIVYCH ZAZNAMU
# ============================================================

results = []

print("\nZacinam kontrolu zaznamu...")

for index, row in manifest.iterrows():

    if (index + 1) % 25 == 0 or index + 1 == len(manifest):
        print(f"Kontroluji zaznam {index + 1}/{len(manifest)}...")

    tic = row["tic_norm"]
    uid = row["uid_norm"]
    label = row["label_norm"]
    sector = row["sector_norm"]
    filename = row["filename_norm"]

    problems = []

    # --------------------------------------------------------
    # 1. Kontrola lokalniho FITS souboru
    # --------------------------------------------------------

    fits_key = (tic, filename.lower())
    matching_fits = local_fits_by_key.get(fits_key, [])

    fits_found = len(matching_fits) > 0

    add_problem(
        problems,
        fits_found,
        "LOCAL_FITS_NOT_FOUND",
    )

    # --------------------------------------------------------
    # 2. Kontrola TIC v targets.csv
    # --------------------------------------------------------

    target_found = tic in targets_by_tic

    add_problem(
        problems,
        target_found,
        "TARGET_NOT_FOUND",
    )

    # --------------------------------------------------------
    # 3. Kontrola odpovidajici svetelne krivky
    # --------------------------------------------------------

    lightcurve_key = (tic, sector, filename)
    matching_lightcurves = lightcurves_by_key.get(lightcurve_key, [])

    lightcurve_found = len(matching_lightcurves) > 0

    add_problem(
        problems,
        lightcurve_found,
        "LIGHTCURVE_NOT_FOUND",
    )

    # --------------------------------------------------------
    # 4. Kontrola konkretniho signalu podle UID
    # --------------------------------------------------------

    matching_signals = signals_by_uid.get(uid, [])

    signal_found = len(matching_signals) > 0

    add_problem(
        problems,
        signal_found,
        "SIGNAL_NOT_FOUND",
    )

    # --------------------------------------------------------
    # 5. Kontrola TIC u konkretniho signalu
    # --------------------------------------------------------

    signal_tic_match = False

    if signal_found:
        signal_tic_match = any(
            signal["tic_norm"] == tic
            for signal in matching_signals
        )

    add_problem(
        problems,
        signal_tic_match,
        "SIGNAL_TIC_MISMATCH",
    )

    # --------------------------------------------------------
    # 6. Kontrola klasifikace podle signals.csv
    # --------------------------------------------------------

    signal_label_match = False

    if signal_found:
        signal_label_match = any(
            signal["tic_norm"] == tic
            and signal["label_norm"] == label
            for signal in matching_signals
        )

    add_problem(
        problems,
        signal_label_match,
        "SIGNAL_LABEL_MISMATCH",
    )

    # --------------------------------------------------------
    # 7. Kontrola UID v lightcurves.csv
    # --------------------------------------------------------

    uid_in_lightcurve = False

    for lightcurve in matching_lightcurves:
        lightcurve_uids = split_semicolon_list(
            lightcurve["signal_uids"]
        )

        if uid in lightcurve_uids:
            uid_in_lightcurve = True
            break

    add_problem(
        problems,
        uid_in_lightcurve,
        "SIGNAL_UID_NOT_IN_LIGHTCURVE",
    )

    # --------------------------------------------------------
    # Vysledek zaznamu
    # --------------------------------------------------------

    results.append({
        "tic_id": tic,
        "signal_uid": uid,
        "manifest_label": row["signal_label"],
        "sector": sector,
        "file_name": filename,
        "fits_found": fits_found,
        "target_found": target_found,
        "lightcurve_found": lightcurve_found,
        "signal_found": signal_found,
        "signal_tic_match": signal_tic_match,
        "signal_label_match": signal_label_match,
        "uid_in_lightcurve": uid_in_lightcurve,
        "successful": len(problems) == 0,
        "problems": ";".join(problems),
    })


# ============================================================
# VYHODNOCENI
# ============================================================

report = pd.DataFrame(results)

errors = report[~report["successful"]].copy()

total = len(report)
successful = int(report["successful"].sum())
failed = total - successful

fits_count = int(report["fits_found"].sum())
targets_count = int(report["target_found"].sum())
lightcurves_count = int(report["lightcurve_found"].sum())
signals_count = int(report["signal_found"].sum())
tic_match_count = int(report["signal_tic_match"].sum())
label_match_count = int(report["signal_label_match"].sum())
uid_match_count = int(report["uid_in_lightcurve"].sum())


# ============================================================
# ULOZENI REPORTU
# ============================================================

report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
errors.to_csv(ERRORS_FILE, index=False, encoding="utf-8-sig")

summary = pd.DataFrame([
    {"kontrola": "Zaznamy v manifestu", "uspesne": total},
    {"kontrola": "Uspesne zaznamy", "uspesne": successful},
    {"kontrola": "Zaznamy s chybami", "uspesne": failed},
    {"kontrola": "Lokalni FITS nalezeny", "uspesne": fits_count},
    {"kontrola": "Targets nalezeny", "uspesne": targets_count},
    {"kontrola": "Lightcurves nalezeny", "uspesne": lightcurves_count},
    {"kontrola": "Signals nalezeny", "uspesne": signals_count},
    {"kontrola": "Shoda TIC", "uspesne": tic_match_count},
    {"kontrola": "Shoda klasifikace v signals.csv", "uspesne": label_match_count},
    {"kontrola": "Shoda UID ve svetelne krivce", "uspesne": uid_match_count},
])

summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# VYPIS VYSLEDKU
# ============================================================

print("\n" + "=" * 70)
print("ROZLOZENI TRID V MANIFESTU")
print("=" * 70)

print(
    manifest["signal_label"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("VYSLEDEK KONTROLY METADAT")
print("=" * 70)

print(f"Zaznamy v manifestu:       {total}")
print(f"Uspešne zaznamy:           {successful}")
print(f"Zaznamy s chybami:         {failed}")
print(f"Lokalni FITS nalezeny:     {fits_count}")
print(f"Targets nalezeny:          {targets_count}")
print(f"Lightcurves nalezeny:      {lightcurves_count}")
print(f"Signals nalezeny:          {signals_count}")
print(f"Shoda TIC:                 {tic_match_count}")
print(f"Shoda klasifikace:         {label_match_count}")
print(f"Shoda UID ve svetelne krivce: {uid_match_count}")

if failed > 0:
    print("\nZAZNAMY S CHYBAMI:")
    print(
        errors[["tic_id", "signal_uid", "manifest_label", "problems"]]
        .to_string(index=False)
    )

print("\nVytvorene reporty:")
print(REPORT_FILE)
print(ERRORS_FILE)
print(SUMMARY_FILE)

print("\nKontrola metadat dokoncena.")