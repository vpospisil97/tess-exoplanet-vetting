"""
Final pretest integrity audit for TESS exoplanet vetting.

Checks dataset and TIC-separated splits, Mantis feature
alignment, and selected checkpoint metadata.

Does not evaluate or train on the test set.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "full_dataset"
VIEWS = DATA / "full_cnn_views.npz"
SPLITS = DATA / "cnn_split_indices.npz"
MANTIS = DATA / "mantis_model_b"
REPORT = ROOT / "docs" / "pretest_audit.md"

COUNTS = {
    "train": 1335,
    "validation": 293,
    "test": 290,
}

SEEDS = (42, 123, 2026)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    print("=" * 70)
    print("TESS - FINAL PRETEST AUDIT")
    print("=" * 70)

    for path in (VIEWS, SPLITS):
        check(path.is_file(), f"Missing file: {path}")

    print("\nChecking dataset...")

    with np.load(VIEWS, allow_pickle=False) as data:
        required = (
            "x_local", "x_global",
            "signal_uid", "tic_key", "y"
        )

        check(
            all(key in data.files for key in required),
            f"Missing dataset arrays. Found: {data.files}",
        )

        local_shape = data["x_local"].shape
        global_shape = data["x_global"].shape
        uid = data["signal_uid"].astype(str)
        tic = data["tic_key"].astype(str)
        labels = data["y"].astype(np.int64)

    check(local_shape == (1918, 101, 2), "Local shape mismatch")
    check(global_shape == (1918, 201, 2), "Global shape mismatch")
    check(uid.shape == (1918,), "UID shape mismatch")
    check(tic.shape == (1918,), "TIC shape mismatch")
    check(labels.shape == (1918,), "Label shape mismatch")
    check(len(set(uid)) == 1918, "Duplicate signal UIDs")
    check(all(uid), "Empty signal UID")
    check(all(tic), "Empty TIC")
    check(np.isin(labels, [0, 1]).all(), "Invalid binary labels")

    print("Dataset: 1918 signals")
    print("Local:", local_shape)
    print("Global:", global_shape)

    print("\nChecking split integrity...")

    with np.load(SPLITS, allow_pickle=False) as data:
        print("Available split keys:", data.files)

        indices = {}

        for name in COUNTS:
            key = f"{name}_idx"

            check(
                key in data.files,
                f"Missing split key {key}",
            )

            indices[name] = np.asarray(
                data[key], dtype=np.int64
            )

    tic_sets = {}
    index_sets = {}

    for name, idx in indices.items():
        check(idx.ndim == 1, f"{name}: invalid dimensions")
        check(len(idx) == COUNTS[name], f"{name}: wrong size")
        check(
            np.all((idx >= 0) & (idx < len(uid))),
            f"{name}: out-of-range indices",
        )
        check(
            len(np.unique(idx)) == len(idx),
            f"{name}: duplicate indices",
        )

        tic_sets[name] = set(tic[idx].tolist())
        index_sets[name] = set(idx.tolist())

        print(
            f"{name}: {len(idx)} signals, "
            f"{len(tic_sets[name])} unique TICs"
        )

    for i, first in enumerate(COUNTS):
        for second in list(COUNTS)[i + 1:]:
            index_overlap = (
                index_sets[first] & index_sets[second]
            )

            tic_overlap = (
                tic_sets[first] & tic_sets[second]
            )

            print(
                f"{first} vs {second}: "
                f"index overlap={len(index_overlap)}, "
                f"TIC overlap={len(tic_overlap)}"
            )

            check(
                not index_overlap,
                f"Index leakage: {first}/{second}",
            )

            check(
                not tic_overlap,
                f"TIC leakage: {first}/{second}",
            )

    all_indices = set().union(*index_sets.values())

    check(
        all_indices == set(range(1918)),
        "Splits do not cover the dataset exactly once",
    )

    print("Split integrity: PASS")

    print("\nChecking Mantis features...")

    for name in ("train", "validation"):
        path = MANTIS / f"features_{name}.npz"

        check(path.is_file(), f"Missing: {path}")

        with np.load(path, allow_pickle=False) as data:
            idx = indices[name]

            check(
                np.array_equal(data["indices"], idx),
                f"{name}: index mismatch",
            )

            check(
                np.array_equal(data["y"], labels[idx]),
                f"{name}: label mismatch",
            )

            check(
                np.array_equal(
                    data["signal_uid"].astype(str), uid[idx]
                ),
                f"{name}: signal UID mismatch",
            )

            check(
                np.array_equal(
                    data["tic_key"].astype(str), tic[idx]
                ),
                f"{name}: TIC mismatch",
            )

            features = data["features"]

            check(
                features.shape == (COUNTS[name], 1024),
                f"{name}: feature shape mismatch",
            )

            check(
                np.isfinite(features).all(),
                f"{name}: non-finite features",
            )

            print(f"{name}: {features.shape}, alignment OK")

    print("Mantis feature alignment: PASS")

    print("\nChecking selected Mantis checkpoints...")

    selection_path = (
        MANTIS / "head_grid_v1"
        / "best_head_config.json"
    )

    check(
        selection_path.is_file(),
        "Missing selected Mantis configuration",
    )

    selection = json.loads(
        selection_path.read_text(encoding="utf-8")
    )

    trial = selection["selected_trial"]
    config = selection["config"]

    check(
        trial == "hidden_128__drop_0p5",
        f"Unexpected selected trial: {trial}",
    )

    check(
        config["hidden_size"] == 128,
        "Hidden size mismatch",
    )

    check(
        config["dropout"] == 0.5,
        "Dropout mismatch",
    )

    check(
        selection.get("test_set_used") is False,
        "Unexpected test-set metadata",
    )

    for seed in SEEDS:
        folder = (
            MANTIS / "head_grid_v1"
            / trial / f"seed_{seed}"
        )

        checkpoint = folder / "best_head.pt"
        metrics_path = folder / "metrics.json"

        check(
            checkpoint.is_file()
            and checkpoint.stat().st_size > 0,
            f"Missing checkpoint: seed {seed}",
        )

        check(
            metrics_path.is_file(),
            f"Missing metrics: seed {seed}",
        )

        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        check(
            metrics["seed"] == seed,
            f"Seed mismatch: {seed}",
        )

        check(
            metrics["config"] == config,
            f"Config mismatch: seed {seed}",
        )

        check(
            metrics.get("test_set_used") is False,
            f"Unexpected test usage: seed {seed}",
        )

        print(f"Seed {seed}: checkpoint metadata OK")

    print("Mantis checkpoint checks: PASS")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# TESS - Pretest Integrity Audit",
        "",
        "## Dataset",
        "",
        "- 1918 labeled binary signals",
        "- Local views: 1918 x 101 x 2",
        "- Global views: 1918 x 201 x 2",
        "",
        "## Splits",
        "",
        "| Split | Signals | Unique TICs |",
        "|---|---:|---:|",
    ]

    for name in COUNTS:
        lines.append(
            f"| {name} | {COUNTS[name]} "
            f"| {len(tic_sets[name])} |"
        )

    lines += [
        "",
        "- No overlapping signal indices",
        "- No overlapping TIC identifiers",
        "- All observations assigned exactly once",
        "",
        "## Model B",
        "",
        "- Training and validation embeddings aligned",
        f"- Selected MLP: {trial}",
        "- Three selected checkpoints found and metadata verified",
        "- Checkpoint tensor contents not loaded in this audit",
        "",
        "## SHA-256 fingerprints",
        "",
        f"- Views: `{sha256(VIEWS)}`",
        f"- Splits: `{sha256(SPLITS)}`",
        "",
        "## Held-out test",
        "",
        "Only split indices and TIC metadata were inspected.",
        "No test predictions or test metrics were calculated.",
        "",
    ]

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("Split integrity: PASS")
    print("TIC leakage: NONE")
    print("Mantis feature alignment: PASS")
    print("Selected Mantis checkpoints: PASS")
    print("Test predictions calculated: NO")
    print("Report:", REPORT)
    print("PRETEST AUDIT COMPLETE")


if __name__ == "__main__":
    main()
