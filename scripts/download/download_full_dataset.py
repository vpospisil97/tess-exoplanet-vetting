"""Stahuje vybrané FITS soubory s logováním a kontrolou disku."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
from astropy.io import fits


BASE = Path(__file__).resolve().parent

MANIFEST = (
    BASE
    / "full_dataset"
    / "fits_download_manifest.csv"
)

OUTPUT = BASE / "full_dataset" / "fits"
PILOT = BASE / "TESS_sample_150"

LOG_FILE = (
    BASE
    / "full_dataset"
    / "download_log.csv"
)

MIN_FREE_GIB = 40
CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 3
TIMEOUT = (20, 120)

LOG_FIELDS = [
    "hf_path",
    "status",
    "size_bytes",
    "message",
]


def parse_args():
    """Načte argumenty pro bezpečné hromadné stahování."""
    parser = argparse.ArgumentParser(
        description="TESS FITS downloader"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximalni pocet souboru pro tento beh. 0 = vse."
    )

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_FREE_GIB,
    )

    return parser.parse_args()


def ensure_log():
    """Připraví soubor protokolu stažených dat."""
    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not LOG_FILE.exists():
        with LOG_FILE.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=LOG_FIELDS
            )
            writer.writeheader()


def log_result(
    hf_path,
    status,
    size_bytes=0,
    message=""
):
    """Zapíše výsledek jednoho pokusu o stažení."""
    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LOG_FIELDS
        )

        writer.writerow({
            "hf_path": hf_path,
            "status": status,
            "size_bytes": size_bytes,
            "message": message[:300],
        })


def validate_fits(path):
    """Ověří strukturu staženého souboru FITS."""
    try:
        with fits.open(
            path,
            memmap=False,
            checksum=False
        ) as hdul:

            if len(hdul) < 2:
                return False

            data = hdul[1].data

            if data is None:
                return False

            names = set(
                hdul[1].columns.names
            )

            if "TIME" not in names:
                return False

            supported_flux = {
                "PDCSAP_FLUX",
                "KSPSAP_FLUX",
                "DET_FLUX",
            }

            if not names.intersection(supported_flux):
                return False

            if len(data) == 0:
                return False

        return True

    except Exception:
        return False


def free_gib():
    """Zjistí aktuální volné místo na pracovním disku v GiB."""
    return (
        shutil.disk_usage(BASE).free
        / (1024 ** 3)
    )


def pilot_file(hf_path):
    """Najde případnou existující kopii souboru z pilotního vzorku."""
    path = PILOT / hf_path

    if path.is_file():
        return path

    return None


def download_one(
    session,
    url,
    destination,
    min_free_gib
):
    """Stáhne a zkontroluje jeden soubor při dodržení diskové rezervy."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = destination.with_name(
        destination.name + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            with session.get(
                url,
                stream=True,
                timeout=TIMEOUT,
                allow_redirects=True
            ) as response:

                response.raise_for_status()

                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        if free_gib() < min_free_gib:
                            raise RuntimeError(
                                "Nedostatek mista na disku"
                            )

                        handle.write(chunk)

            if not validate_fits(temporary):
                raise RuntimeError(
                    "Stazeny soubor neni platny FITS"
                )

            temporary.replace(destination)

            return (
                True,
                destination.stat().st_size,
                "downloaded"
            )

        except Exception as exc:
            if temporary.exists():
                temporary.unlink()

            if (
                "Nedostatek mista"
                in str(exc)
            ):
                raise

            print(
                f"  Pokus {attempt}/{MAX_RETRIES}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 3)

    return (
        False,
        0,
        "download_failed"
    )


def main():
    """Stáhne vybrané FITS a zaznamená úspěchy i chyby."""
    args = parse_args()

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    ensure_log()

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"tic_id": str},
        low_memory=False
    )

    required = {
        "hf_path",
        "download_url",
    }

    if not required.issubset(
        manifest.columns
    ):
        raise ValueError(
            "Manifest nema povinne sloupce."
        )

    if manifest["hf_path"].duplicated().any():
        raise ValueError(
            "Manifest obsahuje duplicitni FITS cesty."
        )

    print("=" * 70)
    print("TESS FULL DATASET DOWNLOADER")
    print("=" * 70)

    print("FITS v manifestu:", len(manifest))
    print(
        "Volne misto:",
        round(free_gib(), 2),
        "GiB"
    )
    print(
        "Minimalni rezerva:",
        args.min_free_gib,
        "GiB"
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": "TESS-DL-Research/1.0"
    })

    processed = 0
    downloaded = 0
    reused = 0
    skipped = 0
    failed = 0

    started = time.time()

    for index, row in manifest.iterrows():
        if (
            args.limit > 0
            and processed >= args.limit
        ):
            break

        hf_path = str(row["hf_path"])
        url = str(row["download_url"])

        relative = Path(hf_path)

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".fits"
            or len(relative.parts) != 2
        ):
            raise ValueError(
                f"Neplatna cesta: {hf_path}"
            )

        destination = OUTPUT / relative

        processed += 1

        print(
            f"\n[{processed}] "
            f"{hf_path}"
        )

        if destination.exists():
            if validate_fits(destination):
                skipped += 1
                print("  Jiz stazeno.")
                continue

            destination.unlink()

        existing_pilot = pilot_file(hf_path)

        if existing_pilot:
            if validate_fits(existing_pilot):
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.copy2(
                    existing_pilot,
                    destination
                )

                reused += 1

                log_result(
                    hf_path,
                    "reused",
                    destination.stat().st_size
                )

                print("  Prevzato z pilotu.")
                continue

        if free_gib() <= args.min_free_gib:
            print(
                "Dosazena minimalni diskova rezerva."
            )
            break

        success, size, message = download_one(
            session,
            url,
            destination,
            args.min_free_gib
        )

        if success:
            downloaded += 1

            log_result(
                hf_path,
                "downloaded",
                size
            )

            print(
                "  Stazeno:",
                round(size / 1024**2, 2),
                "MiB"
            )

        else:
            failed += 1

            log_result(
                hf_path,
                "failed",
                0,
                message
            )

            print("  STAHOVANI SELHALO.")

        if processed % 100 == 0:
            elapsed = (
                time.time() - started
            )

            print("\nPRUBEZNY STAV")
            print("Zpracovano:", processed)
            print("Stazeno:", downloaded)
            print("Prevzato:", reused)
            print("Preskoceno:", skipped)
            print("Chyby:", failed)
            print(
                "Cas:",
                round(elapsed / 60, 1),
                "min"
            )
            print(
                "Volne misto:",
                round(free_gib(), 2),
                "GiB"
            )

    print("\n" + "=" * 70)
    print("KONEC STAHOVANI")
    print("=" * 70)

    print("Zpracovano:", processed)
    print("Stazeno:", downloaded)
    print("Prevzato z pilotu:", reused)
    print("Jiz existovalo:", skipped)
    print("Chyby:", failed)

    print(
        "Volne misto:",
        round(free_gib(), 2),
        "GiB"
    )

    print(
        "Log:",
        LOG_FILE
    )


if __name__ == "__main__":
    main()
