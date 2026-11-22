"""Prověřuje neúspěšné přenosy a poškozené soubory FITS."""

from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import time

import pandas as pd
import requests
from astropy.io import fits

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "full_dataset" / "fits_download_manifest.csv"
REPORT = BASE / "full_dataset" / "failed_download_diagnostics.csv"

FAILED_TICS = {
    "126737992",
    "321068176",
    "382200986",
}

SUPPORTED_FLUX = {
    "PDCSAP_FLUX",
    "KSPSAP_FLUX",
    "DET_FLUX",
}

def inspect_fits(path):
    """Prověří strukturu a čitelnost problematického FITS."""
    with fits.open(path, memmap=False, checksum=False) as hdul:
        if len(hdul) < 2:
            return "invalid", "Missing extension 1", ""

        hdu = hdul[1]
        if not hasattr(hdu, "columns") or hdu.columns is None:
            return "invalid", "Extension 1 has no columns", ""

        columns = set(hdu.columns.names or [])
        flux_columns = sorted(columns & SUPPORTED_FLUX)

        if "TIME" not in columns:
            return "invalid", "TIME column missing", ",".join(flux_columns)

        if not flux_columns:
            return "invalid", "Supported flux column missing", ""

        if hdu.data is None or len(hdu.data) == 0:
            return "invalid", "No data rows", ",".join(flux_columns)

        return "valid", f"{len(hdu.data)} rows", ",".join(flux_columns)


def main():
    """Shrne příčiny vybraných neúspěšných stažení."""
    print("=" * 75)
    print("DIAGNOSTIKA TRI CHYBEJICICH TESS FITS")
    print("=" * 75)

    manifest = pd.read_csv(MANIFEST, dtype={"tic_id": str})

    targets = manifest[
        manifest["hf_path"].astype(str).str.startswith(
            tuple(f"TIC_{tic}/" for tic in FAILED_TICS)
        )
    ].copy()

    # Testujeme jen tri konkretni soubory,
    # nikoli vsechny sektory techto hvezd.
    names = {
        "tess2023043185947-s0062-0000000126737992-0254-a_fast-lc.fits",
        "tess2025258001959-s0097-0000000321068176-0294-a_fast-lc.fits",
        "tess2020212050318-s0028-0000000382200986-0190-s_lc.fits",
    }

    targets = targets[
        targets["hf_path"].astype(str).map(
            lambda value: Path(value).name in names
        )
    ]

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

            print("\n" + "-" * 75)
            print("Soubor:", hf_path)
            print("URL:", url)

            result = {
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "hf_path": hf_path,
                "url": url,
                "http_status": "",
                "final_url": "",
                "content_type": "",
                "bytes_received": 0,
                "fits_status": "not_checked",
                "fits_detail": "",
                "flux_columns": "",
                "error_type": "",
                "error_message": "",
                "elapsed_seconds": 0,
            }

            start = time.monotonic()

            try:
                with TemporaryDirectory() as tmp:
                    destination = Path(tmp) / "test.fits"

                    with session.get(
                        url,
                        stream=True,
                        timeout=(20, 120),
                        allow_redirects=True,
                    ) as response:

                        result["http_status"] = response.status_code
                        result["final_url"] = response.url
                        result["content_type"] = response.headers.get(
                            "Content-Type", ""
                        )

                        print("HTTP:", response.status_code)
                        print("Content-Type:", result["content_type"])

                        response.raise_for_status()

                        with destination.open("wb") as handle:
                            for chunk in response.iter_content(
                                chunk_size=1024 * 1024
                            ):
                                if chunk:
                                    handle.write(chunk)
                                    result["bytes_received"] += len(chunk)

                    print("Prijato bajtu:", result["bytes_received"])

                    try:
                        status, detail, flux = inspect_fits(destination)

                        result["fits_status"] = status
                        result["fits_detail"] = detail
                        result["flux_columns"] = flux

                        print("FITS:", status)
                        print("Detail:", detail)
                        print("Flux:", flux)

                    except Exception as exc:
                        result["fits_status"] = "invalid"
                        result["fits_detail"] = str(exc)
                        print("FITS CHYBA:", exc)

            except requests.exceptions.HTTPError as exc:
                result["error_type"] = "HTTPError"
                result["error_message"] = str(exc)
                print("HTTP CHYBA:", exc)

            except requests.exceptions.RequestException as exc:
                result["error_type"] = type(exc).__name__
                result["error_message"] = str(exc)
                print("SITOVA CHYBA:", exc)

            except Exception as exc:
                result["error_type"] = type(exc).__name__
                result["error_message"] = str(exc)
                print("JINA CHYBA:", exc)

            result["elapsed_seconds"] = round(
                time.monotonic() - start, 2
            )

            results.append(result)

    report = pd.DataFrame(results)
    report.to_csv(REPORT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("SOUHRN DIAGNOSTIKY")
    print("=" * 75)

    print(
        report[
            [
                "hf_path",
                "http_status",
                "bytes_received",
                "fits_status",
                "error_type",
                "fits_detail",
            ]
        ].to_string(index=False)
    )

    print("\nReport:", REPORT)
    print("DIAGNOSTIKA DOKONCENA")


if __name__ == "__main__":
    main()
