"""Měří časové pokrytí tranzitů v dostupných FITS pozorováních."""


from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.io import fits


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

MANIFEST = DATA / "training_manifest.csv"
LINKS = DATA / "training_fits_links.csv"
FITS_ROOT = DATA / "fits"

OUT_DETAIL = DATA / "transit_coverage_detail.csv"
OUT_SUMMARY = DATA / "transit_coverage_summary.csv"

MIN_IN_TRANSIT = 5
MIN_OUT_TRANSIT = 10

BJD_OFFSET = 2457000.0


def normalize_path(value):
    """Sjednotí zápis cest k FITS pro porovnávání souborů."""
    if pd.isna(value):
        return ""

    return str(value).replace("\\", "/").strip().lower()


def find_fits_path(value):
    """Vyhledá lokální FITS odpovídající uvedené cestě."""
    raw = str(value).replace("\\", "/").strip()
    path = Path(raw)

    if path.is_file():
        return path

    if not path.is_absolute():
        candidate = DATA / path

        if candidate.is_file():
            return candidate

        candidate = BASE / path

        if candidate.is_file():
            return candidate

    parts = raw.split("/")

    for index, part in enumerate(parts):
        if part.lower() == "fits":
            relative = Path(*parts[index + 1:])
            candidate = FITS_ROOT / relative

            if candidate.is_file():
                return candidate

    return None


def load_times(path):
    """Načte časová měření z použitelných FITS záznamů."""
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            data = hdu.data

            if data is None or not hasattr(data, "names"):
                continue

            names = {
                str(name).upper(): name
                for name in data.names
            }

            if "TIME" not in names:
                continue

            time = np.asarray(
                data[names["TIME"]],
                dtype=np.float64,
            )

            valid = np.isfinite(time) & (time > 0)

            if "QUALITY" in names:
                quality = np.asarray(
                    data[names["QUALITY"]]
                )

                valid &= quality == 0

            flux_name = next(
                (
                    names[name]
                    for name in (
                        "PDCSAP_FLUX",
                        "KSPSAP_FLUX",
                        "DET_FLUX",
                        "SAP_FLUX",
                    )
                    if name in names
                ),
                None,
            )

            if flux_name is None:
                continue

            flux = np.asarray(
                data[flux_name],
                dtype=np.float64,
            )

            valid &= np.isfinite(flux) & (flux > 0)

            ref_i = hdu.header.get(
                "BJDREFI",
                hdul[0].header.get("BJDREFI"),
            )

            ref_f = hdu.header.get(
                "BJDREFF",
                hdul[0].header.get("BJDREFF", 0.0),
            )

            if ref_i is None:
                raise ValueError("Missing BJDREFI")

            reference = float(ref_i) + float(ref_f)

            if abs(reference - BJD_OFFSET) > 1:
                raise ValueError(
                    f"Unexpected BJD reference: {reference}"
                )

            return time[valid], reference

    raise ValueError("TIME/FLUX table not found")


def count_transit_points(
    times,
    epoch_bjd,
    period,
    duration_hours,
):
    """Spočítá měření během tranzitu a blízká měření mimo něj."""
    if len(times) == 0:
        return 0, 0

    epoch_btjd = epoch_bjd - BJD_OFFSET
    duration_days = duration_hours / 24.0

    phase_days = (
        (times - epoch_btjd + period / 2) % period
    ) - period / 2

    distance = np.abs(phase_days)

    inside = distance <= duration_days / 2

    outside = (
        (distance > duration_days / 2)
        & (distance <= duration_days * 2.5)
    )

    return (
        int(inside.sum()),
        int(outside.sum()),
    )


def main():
    """Vyhodnotí dostatečnost pozorování každého tranzitu."""
    print("=" * 72)
    print("TESS - KONTROLA POKRYTI TRANZITU")
    print("=" * 72)

    for path in (MANIFEST, LINKS):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(
        MANIFEST,
        low_memory=False,
    )

    links = pd.read_csv(
        LINKS,
        low_memory=False,
    )

    ready = (
        manifest["preliminary_ready"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )

    signals = manifest.loc[
        ready
        & manifest["signal_label"].isin(
            ["confirmed", "false_positive"]
        )
    ].copy()

    links["usable_link"] = (
        links["usable_link"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )

    links = links.loc[
        links["usable_link"]
        & links["signal_uid"].isin(
            signals["signal_uid"]
        )
    ].copy()

    print("Signalu:", len(signals))
    print("Vazeb signal-FITS:", len(links))

    links_by_signal = defaultdict(list)

    for row in links.itertuples(index=False):
        links_by_signal[str(row.signal_uid)].append(
            str(row.file)
        )

    cache = {}
    records = []

    for index, row in enumerate(
        signals.itertuples(index=False),
        start=1,
    ):
        signal_uid = str(row.signal_uid)

        period = float(row.period_days)
        epoch = float(row.epoch_bjd)
        duration = float(row.duration_hours)

        total_inside = 0
        total_outside = 0

        files_ok = 0
        files_failed = 0

        seen = set()

        for file_value in links_by_signal.get(
            signal_uid,
            [],
        ):
            key = normalize_path(file_value)

            if key in seen:
                continue

            seen.add(key)

            if key not in cache:
                path = find_fits_path(file_value)

                if path is None:
                    cache[key] = (
                        None,
                        "file_not_found",
                    )
                else:
                    try:
                        times, reference = load_times(path)

                        cache[key] = (
                            times,
                            None,
                        )

                    except Exception as exc:
                        cache[key] = (
                            None,
                            f"{type(exc).__name__}: {exc}",
                        )

            times, error = cache[key]

            if error is not None:
                files_failed += 1
                continue

            inside, outside = count_transit_points(
                times,
                epoch,
                period,
                duration,
            )

            total_inside += inside
            total_outside += outside

            files_ok += 1

        covered = (
            total_inside >= MIN_IN_TRANSIT
            and total_outside >= MIN_OUT_TRANSIT
        )

        records.append(
            {
                "signal_uid": signal_uid,
                "tic_key": row.tic_key,
                "signal_label": row.signal_label,
                "period_days": period,
                "duration_hours": duration,
                "fits_processed": files_ok,
                "fits_failed": files_failed,
                "points_in_transit": total_inside,
                "points_near_transit": total_outside,
                "coverage_pass": covered,
            }
        )

        if (
            index % 100 == 0
            or index == len(signals)
        ):
            print(
                f"Zpracovano {index}/{len(signals)} | "
                f"unikatnich FITS v cache: {len(cache)}",
                flush=True,
            )

    result = pd.DataFrame(records)

    result.to_csv(
        OUT_DETAIL,
        index=False,
    )

    summary = pd.crosstab(
        result["signal_label"],
        result["coverage_pass"],
    )

    summary.to_csv(OUT_SUMMARY)

    print("\n" + "=" * 72)
    print("VYSLEDKY")
    print("=" * 72)

    print("\nPokryti podle tridy:")
    print(summary.to_string())

    print(
        "\nCelkem proslo:",
        int(result["coverage_pass"].sum()),
    )

    print(
        "Celkem neproslo:",
        int((~result["coverage_pass"]).sum()),
    )

    print("\nStatistika bodu uvnitr tranzitu:")

    print(
        result.groupby("signal_label")[
            "points_in_transit"
        ].describe().to_string()
    )

    print("\nFITS soubory s chybou:")
    print(int(result["fits_failed"].sum()))

    print("\nVystupy:")
    print(OUT_DETAIL)
    print(OUT_SUMMARY)

    print("\nHOTOVO")


if __name__ == "__main__":
    main()
