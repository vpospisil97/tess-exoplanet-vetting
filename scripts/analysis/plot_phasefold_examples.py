"""Vykresluje příklady fázovaného tranzitu z pilotních dat."""


from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy.io import fits


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

COVERAGE = DATA / "transit_coverage_detail.csv"
MANIFEST = DATA / "training_manifest.csv"
LINKS = DATA / "training_fits_links.csv"

OUTPUT_PNG = DATA / "pilot_phasefold_6_examples.png"
OUTPUT_CSV = DATA / "pilot_phasefold_6_examples.csv"

BJD_REFERENCE = 2457000.0

N_PER_CLASS = 3
MAX_SCATTER_POINTS = 2000


def as_bool(series):
    """Sjednotí textové příznaky logických hodnot v tabulce."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )


def resolve_fits(value):
    """Najde dostupný FITS soubor z katalogové cesty."""
    raw = str(value).replace("\\", "/").strip()
    path = Path(raw)

    candidates = [path]

    if not path.is_absolute():
        candidates.append(BASE / path)
        candidates.append(DATA / path)

    parts = raw.split("/")

    for index, part in enumerate(parts):
        if part.lower() == "fits":
            relative = Path(*parts[index + 1:])
            candidates.append(DATA / "fits" / relative)
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_lightcurve(path):
    """Načte fotometrická měření a časovou referenci z FITS."""
    with fits.open(path, memmap=True) as hdul:
        primary_header = hdul[0].header

        for hdu in hdul:
            data = hdu.data

            if data is None:
                continue

            names = getattr(data, "names", None)

            if names is None:
                continue

            columns = {
                str(name).upper(): name
                for name in names
            }

            if "TIME" not in columns:
                continue

            flux_column = next(
                (
                    columns[name]
                    for name in (
                        "PDCSAP_FLUX",
                        "KSPSAP_FLUX",
                        "DET_FLUX",
                        "SAP_FLUX",
                    )
                    if name in columns
                ),
                None,
            )

            if flux_column is None:
                continue

            time = np.asarray(
                data[columns["TIME"]],
                dtype=np.float64,
            )

            flux = np.asarray(
                data[flux_column],
                dtype=np.float64,
            )

            quality_ok = np.ones(
                len(time),
                dtype=bool,
            )

            if "QUALITY" in columns:
                quality = np.asarray(
                    data[columns["QUALITY"]]
                )
                quality_ok = quality == 0

            valid = (
                np.isfinite(time)
                & (time > 0)
                & np.isfinite(flux)
                & (flux > 0)
                & quality_ok
            )

            if valid.sum() < 10:
                continue

            reference_i = hdu.header.get(
                "BJDREFI",
                primary_header.get("BJDREFI"),
            )

            reference_f = hdu.header.get(
                "BJDREFF",
                primary_header.get("BJDREFF", 0.0),
            )

            if reference_i is None:
                raise ValueError("Missing BJDREFI")

            reference = (
                float(reference_i)
                + float(reference_f)
            )

            if abs(reference - BJD_REFERENCE) > 1:
                raise ValueError(
                    f"Unexpected time reference: {reference}"
                )

            time = time[valid] + reference
            flux = flux[valid]

            median_flux = np.median(flux)

            if (
                not np.isfinite(median_flux)
                or median_flux <= 0
            ):
                continue

            normalized_flux = flux / median_flux

            return time, normalized_flux

    raise ValueError(
        "No usable TIME and FLUX data"
    )


def phase_fold(time_bjd, epoch_bjd, period, duration):
    """Převede časy měření na fázi tranzitu."""
    phase_days = (
        (time_bjd - epoch_bjd + period / 2)
        % period
    ) - period / 2

    duration_days = duration / 24.0

    return phase_days / duration_days


def select_examples(coverage):
    """Vybere reprezentativní pilotní signály pro ukázky."""
    coverage = coverage.copy()

    coverage["coverage_pass"] = as_bool(
        coverage["coverage_pass"]
    )

    numeric = [
        "period_days",
        "duration_hours",
        "points_in_transit",
    ]

    for column in numeric:
        coverage[column] = pd.to_numeric(
            coverage[column],
            errors="coerce",
        )

    ratio = (
        coverage["duration_hours"]
        / (coverage["period_days"] * 24.0)
    )

    eligible = coverage.loc[
        coverage["coverage_pass"]
        & coverage["period_days"].between(2, 8)
        & (coverage["points_in_transit"] >= 100)
        & (ratio < 0.12)
    ].copy()

    chosen = []
    used_tics = set()

    for label in ("confirmed", "false_positive"):
        subset = eligible.loc[
            eligible["signal_label"] == label
        ].sort_values(
            "points_in_transit",
            ascending=False,
        )

        class_count = 0

        for row in subset.itertuples(index=False):
            tic = str(row.tic_key)

            if tic in used_tics:
                continue

            chosen.append(row._asdict())
            used_tics.add(tic)
            class_count += 1

            if class_count == N_PER_CLASS:
                break

        if class_count < N_PER_CLASS:
            raise RuntimeError(
                f"Not enough examples for {label}: "
                f"{class_count}/{N_PER_CLASS}"
            )

    return pd.DataFrame(chosen)


def build_phase_curve(row, files):
    """Složí fázovou světelnou křivku z dostupných pozorování."""
    all_phases = []
    all_fluxes = []
    successful_files = 0

    seen = set()

    for file_name in files:
        path = resolve_fits(file_name)

        if path is None:
            continue

        path_key = str(path.resolve()).lower()

        if path_key in seen:
            continue

        seen.add(path_key)

        try:
            time, flux = load_lightcurve(path)
        except Exception as exc:
            print(
                f"  FITS skipped: {path.name} - {exc}"
            )
            continue

        phase = phase_fold(
            time,
            float(row["epoch_bjd"]),
            float(row["period_days"]),
            float(row["duration_hours"]),
        )

        within_window = (
            np.isfinite(phase)
            & np.isfinite(flux)
            & (np.abs(phase) <= 3)
        )

        if not within_window.any():
            continue

        all_phases.append(phase[within_window])
        all_fluxes.append(flux[within_window])
        successful_files += 1

    if not all_phases:
        return None

    phases = np.concatenate(all_phases)
    fluxes = np.concatenate(all_fluxes)

    order = np.argsort(phases)

    return (
        phases[order],
        fluxes[order],
        successful_files,
    )


def bin_lightcurve(phases, fluxes):
    """Agreguje fázovaná měření do pravidelných intervalů."""
    edges = np.linspace(-3, 3, 61)

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    medians = np.full(
        len(centers),
        np.nan,
    )

    for index in range(len(centers)):
        if index == len(centers) - 1:
            mask = (
                (phases >= edges[index])
                & (phases <= edges[index + 1])
            )
        else:
            mask = (
                (phases >= edges[index])
                & (phases < edges[index + 1])
            )

        if mask.sum() >= 3:
            medians[index] = np.median(
                fluxes[mask]
            )

    return centers, medians


def main():
    """Vytvoří grafické ukázky fázově složených tranzitů."""
    print("=" * 72)
    print("TESS - PILOTNI PHASE FOLDING")
    print("=" * 72)

    for path in (COVERAGE, MANIFEST, LINKS):
        if not path.exists():
            raise FileNotFoundError(path)

    coverage = pd.read_csv(COVERAGE)
    manifest = pd.read_csv(
        MANIFEST, low_memory=False
    )
    links = pd.read_csv(
        LINKS, low_memory=False
    )

    selected = select_examples(coverage)

    if manifest["signal_uid"].duplicated().any():
        raise ValueError(
            "Duplicate signal_uid in training manifest"
        )

    epochs = manifest.set_index(
        "signal_uid"
    )["epoch_bjd"]

    selected["epoch_bjd"] = selected[
        "signal_uid"
    ].map(epochs)

    if selected["epoch_bjd"].isna().any():
        raise ValueError(
            "Missing epoch for selected signals"
        )

    if (
        pd.to_numeric(
            selected["epoch_bjd"],
            errors="coerce",
        ) < 2400000
    ).any():
        raise ValueError(
            "Epoch is not in expected absolute BJD format"
        )

    links = links.loc[
        as_bool(links["usable_link"])
    ].copy()

    files_by_signal = defaultdict(list)

    for row in links.itertuples(index=False):
        files_by_signal[str(row.signal_uid)].append(
            str(row.file)
        )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9),
        sharex=True,
    )

    records = []

    for index, row in selected.iterrows():
        label = row["signal_label"]

        plot_row = (
            0 if label == "confirmed" else 1
        )

        plot_col = (
            len(records) % N_PER_CLASS
        )

        ax = axes[plot_row, plot_col]

        uid = str(row["signal_uid"])

        print(f"\nProcessing: {uid} ({label})")

        curve = build_phase_curve(
            row,
            files_by_signal.get(uid, []),
        )

        if curve is None:
            ax.text(
                0.5, 0.5,
                "No usable curve",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{uid} - {label}")

            records.append({
                "signal_uid": uid,
                "signal_label": label,
                "status": "no_data",
                "n_points": 0,
                "n_fits": 0,
            })
            continue

        phases, fluxes, n_fits = curve

        if len(phases) > MAX_SCATTER_POINTS:
            indices = np.linspace(
                0,
                len(phases) - 1,
                MAX_SCATTER_POINTS,
                dtype=int,
            )
        else:
            indices = np.arange(len(phases))

        ax.scatter(
            phases[indices],
            fluxes[indices],
            s=3,
            alpha=0.16,
            label="Measurements",
        )

        centers, medians = bin_lightcurve(
            phases,
            fluxes,
        )

        ax.plot(
            centers,
            medians,
            linewidth=2,
            label="Binned median",
        )

        ax.axvspan(
            -0.5,
            0.5,
            alpha=0.10,
        )

        ax.axhline(
            1.0,
            linestyle=":",
            alpha=0.5,
        )

        lower, upper = np.percentile(
            fluxes,
            [1, 99],
        )

        if upper > lower:
            margin = (upper - lower) * 0.12
            ax.set_ylim(
                lower - margin,
                upper + margin,
            )

        ax.set_xlim(-3, 3)
        ax.grid(alpha=0.2)

        ax.set_title(
            f"{uid} | {label}\n"
            f"P={row['period_days']:.2f} d | "
            f"D={row['duration_hours']:.2f} h"
        )

        ax.set_xlabel(
            "Phase / transit duration"
        )
        ax.set_ylabel(
            "Normalized flux"
        )

        ax.legend(fontsize=8)

        records.append({
            "signal_uid": uid,
            "tic_key": row["tic_key"],
            "signal_label": label,
            "period_days": row["period_days"],
            "duration_hours": row["duration_hours"],
            "status": "ok",
            "n_points": len(phases),
            "n_fits": n_fits,
        })

        print(
            f"  FITS: {n_fits} | "
            f"Phase-window points: {len(phases)}"
        )

    fig.suptitle(
        "TESS - Phase-folded transit examples\n"
        "Top: confirmed | Bottom: false positives",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    pd.DataFrame(records).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 72)
    print("HOTOVO")
    print("=" * 72)

    print("\nGraf:")
    print(OUTPUT_PNG)

    print("\nPrehled vybranych signalu:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
