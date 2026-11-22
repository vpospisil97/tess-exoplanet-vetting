"""Zjišťuje HTTP metadata problematických FITS souborů."""

from pathlib import Path
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import time

import pandas as pd
import requests
from astropy.io import fits

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "full_dataset" / "fits_download_manifest.csv"
REPORT = BASE / "full_dataset" / "http_header_diagnostics.csv"

FAILED_FILES = {
    "TIC_126737992/tess2023043185947-s0062-0000000126737992-0254-a_fast-lc.fits",
    "TIC_321068176/tess2025258001959-s0097-0000000321068176-0294-a_fast-lc.fits",
    "TIC_382200986/tess2020212050318-s0028-0000000382200986-0190-s_lc.fits",
}

EXPECTED_SIZES = {
    "TIC_126737992": 11131200,
    "TIC_321068176": 23621760,
    "TIC_382200986": 1840320,
}

TIMEOUT = (20, 120)
CHUNK_SIZE = 1024 * 1024


def headers_to_dict(headers):
    """Vybere diagnostické položky z HTTP hlaviček."""
    return {
        "content_length": headers.get("Content-Length", ""),
        "content_type": headers.get("Content-Type", ""),
        "accept_ranges": headers.get("Accept-Ranges", ""),
        "content_range": headers.get("Content-Range", ""),
        "etag": headers.get("ETag", ""),
        "transfer_encoding": headers.get("Transfer-Encoding", ""),
        "content_encoding": headers.get("Content-Encoding", ""),
    }


def main():
    """Porovná HTTP odpovědi problematických souborů."""
    manifest = pd.read_csv(MANIFEST)
    targets = manifest[
        manifest["hf_path"].isin(FAILED_FILES)
    ].copy()

    if len(targets) != 3:
        raise RuntimeError(
            f"Ocekavany 3 soubory, nalezeno {len(targets)}."
        )

    results = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "TESS-DL-Research/1.0"
        })

        for _, row in targets.iterrows():
            hf_path = str(row["hf_path"])
            url = str(row["download_url"])
            tic = hf_path.split("/")[0]
            expected_size = EXPECTED_SIZES[tic]

            print("\n" + "=" * 75)
            print("SOUBOR:", hf_path)
            print("=" * 75)

            result = {
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "hf_path": hf_path,
                "expected_fits_bytes": expected_size,
                "head_status": "",
                "head_content_length": "",
                "head_accept_ranges": "",
                "head_etag": "",
                "get_status": "",
                "get_content_length": "",
                "get_content_range": "",
                "get_bytes": 0,
                "range_status": "",
                "range_content_range": "",
                "range_bytes": 0,
                "fits_result": "",
                "error": "",
            }

            try:
                print("\n1. HEAD")

                with session.head(
                    url,
                    allow_redirects=True,
                    timeout=TIMEOUT,
                ) as response:
                    h = headers_to_dict(response.headers)

                    result["head_status"] = response.status_code
                    result["head_content_length"] = h["content_length"]
                    result["head_accept_ranges"] = h["accept_ranges"]
                    result["head_etag"] = h["etag"]

                    print("HTTP:", response.status_code)
                    print("Content-Length:", h["content_length"])
                    print("Accept-Ranges:", h["accept_ranges"])
                    print("ETag:", h["etag"])

                print("\n2. GET")

                with TemporaryDirectory() as tmp:
                    destination = Path(tmp) / "test.fits"

                    with session.get(
                        url,
                        stream=True,
                        allow_redirects=True,
                        timeout=TIMEOUT,
                    ) as response:

                        h = headers_to_dict(response.headers)

                        result["get_status"] = response.status_code
                        result["get_content_length"] = h["content_length"]
                        result["get_content_range"] = h["content_range"]

                        print("HTTP:", response.status_code)
                        print("Content-Length:", h["content_length"])
                        print("Content-Range:", h["content_range"])

                        response.raise_for_status()

                        with destination.open("wb") as handle:
                            for chunk in response.iter_content(
                                chunk_size=CHUNK_SIZE
                            ):
                                if chunk:
                                    handle.write(chunk)
                                    result["get_bytes"] += len(chunk)

                    print("Skutecne bajty:", result["get_bytes"])
                    print("FITS ocekava:", expected_size)

                    try:
                        with fits.open(
                            destination,
                            memmap=False,
                            checksum=False,
                        ) as hdul:
                            data = hdul[1].data
                            result["fits_result"] = (
                                f"valid, rows={len(data)}"
                            )

                    except Exception as exc:
                        result["fits_result"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                    print("FITS:", result["fits_result"])

                print("\n3. RANGE - KONEC SOUBORU")

                start_byte = max(0, expected_size - 1024)
                end_byte = expected_size - 1

                with session.get(
                    url,
                    headers={
                        "Range": f"bytes={start_byte}-{end_byte}"
                    },
                    stream=True,
                    allow_redirects=True,
                    timeout=TIMEOUT,
                ) as response:

                    result["range_status"] = response.status_code
                    result["range_content_range"] = response.headers.get(
                        "Content-Range", ""
                    )

                    print("HTTP:", response.status_code)
                    print(
                        "Content-Range:",
                        result["range_content_range"]
                    )

                    # Nestahujeme cele telo, pokud server
                    # ignoruje Range a vraci HTTP 200.
                    if response.status_code == 206:
                        for chunk in response.iter_content(
                            chunk_size=4096
                        ):
                            result["range_bytes"] += len(chunk)

                    elif response.status_code == 416:
                        print("Pozadovany rozsah neni dostupny.")

                    else:
                        print(
                            "Server nevratil partial content; "
                            "telo nestahujeme."
                        )

                    print("Range bajty:", result["range_bytes"])

            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                print("CHYBA:", result["error"])

            results.append(result)

    report = pd.DataFrame(results)
    report.to_csv(
        REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 75)
    print("SOUHRN")
    print("=" * 75)

    columns = [
        "hf_path",
        "head_status",
        "head_content_length",
        "get_status",
        "get_content_length",
        "get_bytes",
        "expected_fits_bytes",
        "range_status",
        "range_content_range",
    ]

    print(report[columns].to_string(index=False))
    print("\nReport:", REPORT)
    print("DIAGNOSTIKA DOKONCENA")


if __name__ == "__main__":
    main()
