
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy.io import fits


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

BJD_REF = 2457000.0

N_GLOBAL = 201
N_LOCAL = 101
MIN_PER_BIN = 3


def boolean(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
        .fillna(False)
    )


def find_file(value):
    raw = str(value).strip().replace("\\", "/")
    path = Path(raw)

    options = [path]

    if not path.is_absolute():
        options += [
            BASE / path,
            DATA / path,
        ]

    parts = raw.split("/")

    for i, part in enumerate(parts):
        if part.lower() == "fits":
            options.append(
                DATA / "fits" / Path(*parts[i + 1:])
            )
            break

    return next(
        (p for p in options if p.is_file()),
        None,
    )


def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            data = hdu.data

            names = getattr(data, "names", None)

            if names is None:
                continue

            cols = {
                str(k).upper(): k
                for k in names
            }

            if "TIME" not in cols:
                continue

            flux_col = next(
                (
                    cols[k]
                    for k in (
                        "PDCSAP_FLUX",
                        "KSPSAP_FLUX",
                        "DET_FLUX",
                        "SAP_FLUX",
                    )
                    if k in cols
                ),
                None,
            )

            if flux_col is None:
                continue

            time = np.asarray(
                data[cols["TIME"]],
                dtype=np.float64,
            )

            flux = np.asarray(
                data[flux_col],
                dtype=np.float64,
            )

            keep = (
                np.isfinite(time)
                & (time > 0)
                & np.isfinite(flux)
                & (flux > 0)
            )

            if "QUALITY" in cols:
                keep &= (
                    np.asarray(data[cols["QUALITY"]]) == 0
                )

            if keep.sum() < 30:
                continue

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

            if abs(reference - BJD_REF) > 1:
                raise ValueError(
                    f"Unexpected BJD reference: {reference}"
                )

            return (
                time[keep] + reference,
                flux[keep],
            )

    raise ValueError("No usable light curve")


def get_phases(time, epoch, period):
    phase_days = (
        (time - epoch + period / 2) % period
    ) - period / 2

    return phase_days


def bin_median(x, y, left, right, bins):
    edges = np.linspace(
        left,
        right,
        bins + 1,
    )

    indices = (
        np.searchsorted(
            edges,
            x,
            side="right",
        ) - 1
    )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & (indices >= 0)
        & (indices < bins)
    )

    indices = indices[good]
    y = y[good]

    values = np.zeros(
        bins,
        dtype=np.float32,
    )

    coverage = np.zeros(
        bins,
        dtype=np.float32,
    )

    if len(indices) == 0:
        return np.stack(
            [values, coverage],
            axis=-1,
        )

    ordering = np.argsort(
        indices,
        kind="stable",
    )

    indices = indices[ordering]
    y = y[ordering]

    starts = np.flatnonzero(
        np.r_[
            True,
            indices[1:] != indices[:-1],
        ]
    )

    ends = np.r_[
        starts[1:],
        len(indices),
    ]

    for start, end in zip(starts, ends):
        if end - start >= MIN_PER_BIN:
            ix = indices[start]

            values[ix] = np.float32(
                np.clip(
                    np.median(y[start:end]),
                    -0.25,
                    0.25,
                )
            )

            coverage[ix] = 1.0

    return np.stack(
        [values, coverage],
        axis=-1,
    )


def make_example(row, filenames):
    period = float(row["period_days"])
    epoch = float(row["epoch_bjd"])

    duration = (
        float(row["duration_hours"]) / 24
    )

    if not (
        np.isfinite(period)
        and period > 0
        and np.isfinite(epoch)
        and 2400000 < epoch < 2500000
        and np.isfinite(duration)
        and 0 < duration < period / 4
    ):
        return None, "invalid_ephemeris"

    all_time = []
    all_residual = []

    seen = set()

    success = 0
    errors = 0

    for file in filenames:
        path = find_file(file)

        if path is None:
            errors += 1
            continue

        key = str(path.resolve()).lower()

        if key in seen:
            continue

        seen.add(key)

        try:
            time, flux = load_fits(path)

        except Exception:
            errors += 1
            continue

        phase_days = get_phases(
            time,
            epoch,
            period,
        )

        out_of_transit = (
            np.abs(phase_days) > duration
        )

        if out_of_transit.sum() >= 30:
            base_flux = np.median(
                flux[out_of_transit]
            )
        else:
            base_flux = np.median(flux)

        if (
            not np.isfinite(base_flux)
            or base_flux <= 0
        ):
            errors += 1
            continue

        residual = flux / base_flux - 1

        # Exclude enormous instrumental positive
        # outliers; preserve deep eclipses.
        valid = (
            np.isfinite(residual)
            & (residual > -0.999)
            & (residual < 1.0)
        )

        if valid.sum() < 30:
            errors += 1
            continue

        all_time.append(time[valid])
        all_residual.append(residual[valid])

        success += 1

    if not all_time:
        return None, "no_valid_fits"

    times = np.concatenate(all_time)
    residual = np.concatenate(all_residual)

    # Do not count the same cadence twice
    # when multiple FITS products overlap.
    order = np.argsort(times)

    times = times[order]
    residual = residual[order]

    rounded_seconds = np.rint(
        times * 86400
    ).astype(np.int64)

    unique = np.r_[
        True,
        rounded_seconds[1:] != rounded_seconds[:-1],
    ]

    times = times[unique]
    residual = residual[unique]

    phase_days = get_phases(
        times,
        epoch,
        period,
    )

    local = bin_median(
        phase_days / duration,
        residual,
        -2.5,
        2.5,
        N_LOCAL,
    )

    glob = bin_median(
        phase_days / period,
        residual,
        -0.5,
        0.5,
        N_GLOBAL,
    )

    local_fill = float(
        local[:, 1].mean()
    )

    global_fill = float(
        glob[:, 1].mean()
    )

    central = local[40:61, 1].sum()

    if (
        local_fill < 0.45
        or global_fill < 0.30
        or central < 5
    ):
        return (
            None,
            (
                "insufficient_bins: "
                f"local={local_fill:.2f}, "
                f"global={global_fill:.2f}, "
                f"center={central:.0f}"
            ),
        )

    return {
        "local": local,
        "global": glob,
        "files_used": success,
        "files_failed": errors,
        "unique_measurements": len(times),
        "local_bin_fraction": round(
            local_fill, 4
        ),
        "global_bin_fraction": round(
            global_fill, 4
        ),
    }, "ok"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare two-view pilot inputs "
            "from TESS phase-folded FITS."
        )
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=60,
        help=(
            "Number selected per class. "
            "0 = all covered signals."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.per_class < 0:
        parser.error(
            "--per-class must be >= 0"
        )

    for name in (
        "training_manifest.csv",
        "training_fits_links.csv",
        "transit_coverage_detail.csv",
    ):
        if not (DATA / name).exists():
            raise FileNotFoundError(
                DATA / name
            )

    meta = pd.read_csv(
        DATA / "training_manifest.csv",
        low_memory=False,
    )

    links = pd.read_csv(
        DATA / "training_fits_links.csv",
        low_memory=False,
    )

    coverage = pd.read_csv(
        DATA / "transit_coverage_detail.csv",
        low_memory=False,
    )

    if meta["signal_uid"].duplicated().any():
        raise ValueError(
            "Duplicate signal_uid in manifest"
        )

    covered = coverage.loc[
        boolean(coverage["coverage_pass"])
        & coverage["signal_label"].isin(
            ["confirmed", "false_positive"]
        )
    ].copy()

    # CSV contains period/duration already;
    # merge only epoch.
    covered = covered.merge(
        meta[["signal_uid", "epoch_bjd"]],
        on="signal_uid",
        how="left",
        validate="one_to_one",
    )

    covered["duration_period_ratio"] = (
        covered["duration_hours"]
        / (24 * covered["period_days"])
    )

    unusual = (
        covered["duration_period_ratio"] >= 0.25
    )

    print(
        "Coverage passed:",
        len(covered),
    )

    print(
        "Unusual duration-to-period ratio "
        "(excluded for this pilot):",
        int(unusual.sum()),
    )

    covered = covered.loc[
        ~unusual
    ].copy()

    rng = np.random.default_rng(
        args.seed
    )

    if args.per_class:
        picks = []

        for label in (
            "confirmed",
            "false_positive",
        ):
            class_rows = covered.loc[
                covered["signal_label"] == label
            ]

            take = min(
                args.per_class,
                len(class_rows),
            )

            indices = rng.choice(
                class_rows.index.to_numpy(),
                size=take,
                replace=False,
            )

            picks.append(
                class_rows.loc[indices]
            )

        selected = pd.concat(
            picks,
            ignore_index=True,
        )

    else:
        selected = (
            covered.sort_values(
                ["signal_label", "signal_uid"]
            )
            .reset_index(drop=True)
        )

    links = links.loc[
        boolean(links["usable_link"])
    ]

    files_by_signal = defaultdict(list)

    for signal_uid, file in zip(
        links["signal_uid"],
        links["file"],
    ):
        files_by_signal[
            str(signal_uid)
        ].append(str(file))

    views_local = []
    views_global = []

    labels = []
    signal_uids = []
    tics = []

    records = []

    for count, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        signal_uid = str(
            row["signal_uid"]
        )

        example, status = make_example(
            row,
            files_by_signal.get(
                signal_uid,
                [],
            ),
        )

        record = {
            "signal_uid": signal_uid,
            "tic_key": str(row["tic_key"]),
            "signal_label": row["signal_label"],
            "status": status,
        }

        if example is not None:
            views_local.append(
                example["local"]
            )

            views_global.append(
                example["global"]
            )

            labels.append(
                1
                if row["signal_label"] == "confirmed"
                else 0
            )

            signal_uids.append(
                signal_uid
            )

            tics.append(
                str(row["tic_key"])
            )

            for k in (
                "files_used",
                "files_failed",
                "unique_measurements",
                "local_bin_fraction",
                "global_bin_fraction",
            ):
                record[k] = example[k]

        records.append(record)

        if (
            count % 20 == 0
            or count == len(selected)
        ):
            print(
                f"Processed {count}/{len(selected)} "
                f"| usable {len(labels)}",
                flush=True,
            )

    prefix = (
        "pilot"
        if args.per_class
        else "full"
    )

    quality_file = (
        DATA / f"{prefix}_cnn_qc.csv"
    )

    pd.DataFrame(records).to_csv(
        quality_file,
        index=False,
    )

    if not labels:
        raise RuntimeError(
            "No usable signals. "
            f"Check {quality_file}"
        )

    data_path = (
        DATA / f"{prefix}_cnn_views.npz"
    )

    np.savez_compressed(
        data_path,
        x_local=np.stack(
            views_local
        ).astype(np.float32),
        x_global=np.stack(
            views_global
        ).astype(np.float32),
        y=np.asarray(
            labels,
            dtype=np.int64,
        ),
        signal_uid=np.asarray(
            signal_uids,
            dtype=str,
        ),
        tic_key=np.asarray(
            tics,
            dtype=str,
        ),
    )

    preview = (
        DATA / f"{prefix}_cnn_preview.png"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
    )

    for plot_row, (
        label,
        class_name,
    ) in enumerate(
        (
            (1, "confirmed"),
            (0, "false_positive"),
        )
    ):
        matching = np.flatnonzero(
            np.asarray(labels) == label
        )

        if len(matching) == 0:
            for ax in axes[plot_row]:
                ax.text(
                    0.5,
                    0.5,
                    f"No {class_name} signal",
                    transform=ax.transAxes,
                    ha="center",
                )
            continue

        j = int(matching[0])

        for plot_col, (
            view,
            start,
            end,
            view_name,
        ) in enumerate(
            (
                (
                    views_local[j],
                    -2.5,
                    2.5,
                    "local",
                ),
                (
                    views_global[j],
                    -0.5,
                    0.5,
                    "global",
                ),
            )
        ):
            ax = axes[
                plot_row,
                plot_col,
            ]

            x = np.linspace(
                start,
                end,
                len(view),
            )

            observed = (
                view[:, 1] > 0
            )

            ax.plot(
                x[observed],
                view[observed, 0] * 1000,
                "o-",
                ms=2.5,
                lw=1,
            )

            ax.axhline(
                0,
                ls=":",
                color="gray",
                lw=1,
            )

            if view_name == "local":
                ax.axvspan(
                    -0.5,
                    0.5,
                    alpha=0.09,
                )

            ax.set_title(
                f"{signal_uids[j]} | "
                f"{class_name} | "
                f"{view_name}"
            )

            if view_name == "local":
                ax.set_xlabel(
                    "Phase / transit duration"
                )
            else:
                ax.set_xlabel(
                    "Orbital phase"
                )

            ax.set_ylabel(
                "Relative flux (ppt)"
            )

            ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        preview,
        dpi=160,
    )

    plt.close(fig)

    print("\nSUMMARY")

    print(
        pd.DataFrame(records)
        .groupby(
            ["signal_label", "status"]
        )
        .size()
        .to_string()
    )

    print(
        "Usable:",
        len(labels),
    )

    print(
        "Local shape:",
        (len(labels), N_LOCAL, 2),
    )

    print(
        "Global shape:",
        (len(labels), N_GLOBAL, 2),
    )

    print(
        "Dataset:",
        data_path,
    )

    print(
        "QC report:",
        quality_file,
    )

    print(
        "Preview:",
        preview,
    )

    print("DONE")


if __name__ == "__main__":
    main()
