
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

INPUT = DATA / "full_cnn_views.npz"

OUT_MANIFEST = DATA / "cnn_split_manifest.csv"
OUT_INDICES = DATA / "cnn_split_indices.npz"
OUT_SUMMARY = DATA / "cnn_split_summary.csv"

SPLITS = ("train", "validation", "test")
FRACTIONS = (0.70, 0.15, 0.15)


def allocate_groups(groups, rng):
    """
    Rozdeli cele TIC hvezdy do tri mnozin.
    Priblizne zachovava pomer trid.
    """
    result = {}

    grouped = groups.groupby(
        "stratum",
        sort=True,
    )

    for stratum, subset in grouped:
        tic_ids = subset.index.to_numpy(
            dtype=str
        )

        rng.shuffle(tic_ids)

        n = len(tic_ids)

        n_train = int(
            round(n * FRACTIONS[0])
        )

        n_validation = int(
            round(n * FRACTIONS[1])
        )

        n_train = min(n_train, n)

        n_validation = min(
            n_validation,
            n - n_train,
        )

        parts = (
            tic_ids[:n_train],
            tic_ids[
                n_train:n_train + n_validation
            ],
            tic_ids[
                n_train + n_validation:
            ],
        )

        for split_name, part in zip(
            SPLITS,
            parts,
        ):
            for tic in part:
                if tic in result:
                    raise ValueError(
                        f"Duplicate TIC: {tic}"
                    )

                result[tic] = split_name

        print(
            f"Group {stratum}: {n} TIC | "
            f"train {len(parts[0])} | "
            f"validation {len(parts[1])} | "
            f"test {len(parts[2])}"
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TESS train/validation/test "
            "split grouped by TIC ID"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TESS - ROZDELENI DATASETU PODLE TIC")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    print("\nNacitam:", INPUT)

    with np.load(
        INPUT,
        allow_pickle=False,
    ) as data:
        required = {
            "x_local",
            "x_global",
            "y",
            "tic_key",
            "signal_uid",
        }

        missing = required - set(
            data.files
        )

        if missing:
            raise ValueError(
                f"Missing arrays: {sorted(missing)}"
            )

        x_local = data["x_local"]
        x_global = data["x_global"]
        y = data["y"]

        tics = data["tic_key"].astype(str)
        uids = data["signal_uid"].astype(str)

    n = len(y)

    print("\nKONTROLA DAT")

    print("Signalu:", n)
    print("Local shape:", x_local.shape)
    print("Global shape:", x_global.shape)

    if n == 0:
        raise ValueError("Empty dataset")

    if x_local.shape != (n, 101, 2):
        raise ValueError(
            f"Invalid local shape: {x_local.shape}"
        )

    if x_global.shape != (n, 201, 2):
        raise ValueError(
            f"Invalid global shape: {x_global.shape}"
        )

    if not (
        len(tics) == len(uids) == n
    ):
        raise ValueError(
            "Different array lengths"
        )

    if not np.all(
        np.isin(y, [0, 1])
    ):
        raise ValueError(
            "Invalid class labels"
        )

    if not np.isfinite(x_local).all():
        raise ValueError(
            "Local inputs contain NaN or infinity"
        )

    if not np.isfinite(x_global).all():
        raise ValueError(
            "Global inputs contain NaN or infinity"
        )

    if not np.isin(
        x_local[:, :, 1],
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "Invalid local coverage mask"
        )

    if not np.isin(
        x_global[:, :, 1],
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "Invalid global coverage mask"
        )

    tics = np.char.strip(tics)
    uids = np.char.strip(uids)

    if any(
        not tic
        or tic.lower() in ("nan", "none")
        for tic in tics
    ):
        raise ValueError(
            "Invalid TIC ID"
        )

    if any(not uid for uid in uids):
        raise ValueError(
            "Empty signal UID"
        )

    if len(set(uids)) != n:
        raise ValueError(
            "Duplicate signal UID"
        )

    df = pd.DataFrame({
        "npz_index": np.arange(n),
        "signal_uid": uids,
        "tic_key": tics,
        "y": y.astype(int),
        "signal_label": np.where(
            y == 1,
            "confirmed",
            "false_positive",
        ),
        "local_bin_fraction": (
            x_local[:, :, 1].mean(axis=1)
        ),
        "global_bin_fraction": (
            x_global[:, :, 1].mean(axis=1)
        ),
    })

    print(
        "Unikatnich TIC:",
        df["tic_key"].nunique(),
    )

    print("\nROZLOZENI TRID")

    print(
        df["signal_label"]
        .value_counts()
        .to_string()
    )

    # Jedna hvezda muze mit vice signalu.
    # Dokonce muze obsahovat signaly
    # confirmed i false_positive.

    groups = (
        df.groupby("tic_key")["y"]
        .agg(["count", "sum"])
    )

    groups["stratum"] = np.select(
        [
            groups["sum"] == 0,
            groups["sum"] == groups["count"],
        ],
        [
            "negative_only",
            "positive_only",
        ],
        default="mixed",
    )

    print("\nROZDELENI HVEZD")

    rng = np.random.default_rng(
        args.seed
    )

    tic_to_split = allocate_groups(
        groups,
        rng,
    )

    df["split"] = df["tic_key"].map(
        tic_to_split
    )

    if df["split"].isna().any():
        raise AssertionError(
            "Unassigned TIC found"
        )

    # Bezpecnostni kontrola:
    # zadna hvezda nesmi byt
    # soucasne ve dvou mnozinach.

    tic_sets = {
        split: set(
            df.loc[
                df["split"] == split,
                "tic_key",
            ]
        )
        for split in SPLITS
    }

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = (
                tic_sets[split_a]
                & tic_sets[split_b]
            )

            if overlap:
                raise AssertionError(
                    f"TIC leakage: "
                    f"{split_a} vs {split_b}"
                )

    print(
        "\nKontrola TIC leakage: OK"
    )

    # Ukladame indexy do puvodniho NPZ.
    # Nepotrebujeme kopirovat vsechna data.

    index_arrays = {
        f"{split}_idx": df.loc[
            df["split"] == split,
            "npz_index",
        ].to_numpy(dtype=np.int64)
        for split in SPLITS
    }

    all_indices = np.concatenate(
        list(index_arrays.values())
    )

    if (
        len(all_indices) != n
        or len(np.unique(all_indices)) != n
    ):
        raise AssertionError(
            "Duplicate or missing example index"
        )

    df.to_csv(
        OUT_MANIFEST,
        index=False,
    )

    np.savez_compressed(
        OUT_INDICES,
        **index_arrays,
        seed=np.int64(args.seed),
    )

    rows = []

    for split in SPLITS:
        part = df.loc[
            df["split"] == split
        ]

        positive = int(
            (part["y"] == 1).sum()
        )

        negative = int(
            (part["y"] == 0).sum()
        )

        rows.append({
            "split": split,
            "signals": len(part),
            "tics": part["tic_key"].nunique(),
            "confirmed": positive,
            "false_positive": negative,
            "positive_fraction": (
                positive / len(part)
                if len(part)
                else np.nan
            ),
            "local_fill_median": (
                part["local_bin_fraction"].median()
                if len(part)
                else np.nan
            ),
            "global_fill_median": (
                part["global_bin_fraction"].median()
                if len(part)
                else np.nan
            ),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print("\n" + "=" * 70)
    print("VYSLEDKY ROZDELENI")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nKontrola:")
    print(
        "TIC overlap mezi mnozinami: 0"
    )

    print(
        "Celkem signalu:",
        int(summary["signals"].sum()),
    )

    print(
        "Celkem unikatnich TIC:",
        int(summary["tics"].sum()),
    )

    print("\nVYSTUPNI SOUBORY")
    print(OUT_MANIFEST)
    print(OUT_INDICES)
    print(OUT_SUMMARY)

    print("\nHOTOVO")


if __name__ == "__main__":
    main()
