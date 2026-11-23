"""E3: TIC-disjoint five-fold CV setup, validation, and final summary.

Usage from the repository root:
    python -u scripts/training/prepare_e3_folds.py
    python -u scripts/training/prepare_e3_folds.py --report

All new files are written under full_dataset/e3_cross_validation/ or docs/e3_cross_validation.md.
The original 290 held-out test signals are never assigned to a CV fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "full_dataset"
SOURCE = DATA / "full_cnn_views.npz"
ORIGINAL_SPLIT = DATA / "cnn_split_indices.npz"
OUTPUT = DATA / "e3_cross_validation"
FOLD_FILE = OUTPUT / "folds.npz"
MANIFEST = OUTPUT / "folds_manifest.json"
FOLD_COUNT = 5
SEEDS = (42, 123, 2026)
METRICS = ("average_precision", "roc_auc", "balanced_accuracy", "f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source() -> dict:
    for path in (SOURCE, ORIGINAL_SPLIT):
        if not path.is_file():
            raise FileNotFoundError(f"Original local data missing: {path}")

    with np.load(SOURCE, allow_pickle=False) as z:
        required = {"x_local", "x_global", "y", "tic_key", "signal_uid"}
        if not required.issubset(z.files):
            raise ValueError(f"Missing arrays in source: {sorted(required - set(z.files))}")
        local = np.asarray(z["x_local"], dtype=np.float32)
        global_view = np.asarray(z["x_global"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        tic = np.char.strip(np.asarray(z["tic_key"]).astype(str))
        uid = np.char.strip(np.asarray(z["signal_uid"]).astype(str))

    with np.load(ORIGINAL_SPLIT, allow_pickle=False) as z:
        tr = np.asarray(z["train_idx"], dtype=np.int64)
        va = np.asarray(z["validation_idx"], dtype=np.int64)
        te = np.asarray(z["test_idx"], dtype=np.int64)

    n = len(y)
    if n != 1918 or (len(tr), len(va), len(te)) != (1335, 293, 290):
        raise ValueError(f"Unexpected dataset/split counts: {n}, {len(tr)}, {len(va)}, {len(te)}")
    if local.shape != (n, 101, 2) or global_view.shape != (n, 201, 2):
        raise ValueError(f"Unexpected light-curve dimensions: {local.shape}, {global_view.shape}")
    if not (np.isfinite(local).all() and np.isfinite(global_view).all()):
        raise ValueError("Non-finite input values")
    if not np.isin(y, (0, 1)).all() or len(set(uid.tolist())) != n:
        raise ValueError("Unexpected labels or duplicate signal UIDs")
    if any(not t or t.lower() in ("nan", "none") for t in tic):
        raise ValueError("Invalid TIC identifiers")
    if not np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(n)):
        raise ValueError("Original split is incomplete or overlapping")
    for a, b in ((tr, va), (tr, te), (va, te)):
        if set(tic[a]) & set(tic[b]):
            raise ValueError("TIC leakage in original split")

    development = np.sort(np.concatenate([tr, va]))
    if set(tic[development]) & set(tic[te]):
        raise ValueError("Development and held-out test TIC overlap")
    return {"local": local, "global": global_view, "y": y, "tic": tic,
            "uid": uid, "train": tr, "validation": va, "test": te,
            "dev": development}


def verify_folds(data: dict, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    test = set(data["test"].tolist())
    dev = set(data["dev"].tolist())
    seen = []
    for k, (train, val) in enumerate(folds, 1):
        atr, ava = set(train.tolist()), set(val.tolist())
        if not atr or not ava or atr & ava or atr | ava != dev:
            raise ValueError(f"Fold {k}: invalid train/validation partition")
        if atr & test or ava & test:
            raise ValueError(f"Fold {k}: test index leakage")
        if set(data["tic"][train]) & set(data["tic"][val]):
            raise ValueError(f"Fold {k}: TIC leakage")
        if set(data["tic"][val]) & set(data["tic"][data["test"]]):
            raise ValueError(f"Fold {k}: held-out TIC leakage")
        for indices in (train, val):
            if len(np.unique(data["y"][indices])) != 2:
                raise ValueError(f"Fold {k}: one class missing")
        seen.extend(val.tolist())
    if len(seen) != len(dev) or set(seen) != dev or len(set(seen)) != len(seen):
        raise ValueError("Not every development signal appears exactly once as outer validation")


def create_folds() -> dict:
    data = read_source()
    rng = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=42)
    dev = data["dev"]
    folds = [
        (dev[train_pos], dev[val_pos])
        for train_pos, val_pos in rng.split(np.zeros(len(dev)), data["y"][dev], data["tic"][dev])
    ]
    verify_folds(data, folds)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    counts = []
    for k, (tr, va) in enumerate(folds, 1):
        arrays[f"fold_{k}_train_idx"] = tr
        arrays[f"fold_{k}_validation_idx"] = va
        counts.append({"fold": k, "train_signals": len(tr), "validation_signals": len(va),
                       "train_tic": len(set(data["tic"][tr])), "validation_tic": len(set(data["tic"][va])),
                       "validation_confirmed": int((data["y"][va] == 1).sum()),
                       "validation_false_positive": int((data["y"][va] == 0).sum())})
    hashes = {"source_npz": sha256(SOURCE), "original_split_npz": sha256(ORIGINAL_SPLIT)}
    meta = {"experiment": "E3", "method": "StratifiedGroupKFold", "n_splits": FOLD_COUNT,
            "fold_random_state": 42, "development_samples": len(dev),
            "held_out_test_samples": len(data["test"]), "test_used_for_training": False,
            "hashes": hashes, "folds": counts,
            "note": "Previously selected hyperparameters used original validation subset; E3 is supplementary robustness analysis."}
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old != meta:
            raise RuntimeError("Existing E3 fold manifest differs. Refusing to overwrite it.")
        with np.load(FOLD_FILE, allow_pickle=False) as z:
            for key, values in arrays.items():
                if key not in z or not np.array_equal(values, z[key]):
                    raise RuntimeError("Existing E3 folds differ. Refusing to overwrite them.")
    else:
        if FOLD_FILE.exists():
            raise RuntimeError("Existing fold file without manifest: inspect before proceeding")
        np.savez_compressed(FOLD_FILE, **arrays)
        MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("E3 folds verified; original test untouched")
    for x in counts:
        print(f"Fold {x['fold']}: train={x['train_signals']} val={x['validation_signals']} "
              f"TIC={x['train_tic']}/{x['validation_tic']} "
              f"val confirmed/FP={x['validation_confirmed']}/{x['validation_false_positive']}")
    print("Saved:", FOLD_FILE)
    return data


def load_folds(data: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    if not (FOLD_FILE.exists() and MANIFEST.exists()):
        raise FileNotFoundError("Run prepare_e3_folds.py first")
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if meta.get("hashes") != {"source_npz": sha256(SOURCE), "original_split_npz": sha256(ORIGINAL_SPLIT)}:
        raise RuntimeError("Original inputs changed after fold creation")
    with np.load(FOLD_FILE, allow_pickle=False) as z:
        folds = [(np.asarray(z[f"fold_{k}_train_idx"], dtype=np.int64),
                  np.asarray(z[f"fold_{k}_validation_idx"], dtype=np.int64))
                 for k in range(1, FOLD_COUNT + 1)]
    verify_folds(data, folds)
    return folds


def inner_group_split(data: dict, outer_train: np.ndarray, seed: int, fold: int):
    """A fresh inner TIC-separated validation partition, never using outer validation."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold * 101)
    tr, va = next(cv.split(np.zeros(len(outer_train)), data["y"][outer_train], data["tic"][outer_train]))
    inner_train, inner_val = outer_train[tr], outer_train[va]
    if set(data["tic"][inner_train]) & set(data["tic"][inner_val]):
        raise RuntimeError("Inner TIC leakage")
    if any(len(np.unique(data["y"][idx])) != 2 for idx in (inner_train, inner_val)):
        raise RuntimeError("Inner split lacks a class")
    return inner_train, inner_val


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 balanced_accuracy_score, f1_score, confusion_matrix)
    pred = (proba >= 0.5).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"average_precision": float(average_precision_score(y, proba)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def save_run(model: str, fold: int, seed: int | None, result: dict, indices, data, probas):
    name = f"fold_{fold:02d}_seed_{seed}" if seed is not None else f"fold_{fold:02d}"
    folder = OUTPUT / model
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{name}.json"
    predictions = folder / f"{name}_predictions.csv"
    if destination.exists() or predictions.exists():
        raise FileExistsError(f"Refusing to overwrite existing E3 result: {destination}")
    result.update({"model": model, "fold": fold, "seed": seed, "test_set_used": False,
                   "outer_validation_signals": len(indices), "outer_validation_tic": len(set(data["tic"][indices]))})
    # Write prediction file before JSON so the latter acts as the completion marker.
    import csv
    with predictions.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("npz_index", "signal_uid", "tic_key", "true_label", "predicted_score"))
        for idx, p in zip(indices, probas):
            w.writerow((int(idx), data["uid"][idx], data["tic"][idx], int(data["y"][idx]), float(p)))
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}: AP={result['metrics']['average_precision']:.4f} "
          f"ROC={result['metrics']['roc_auc']:.4f} F1={result['metrics']['f1']:.4f}", flush=True)


def load_complete(model: str, fold: int, seed: int | None):
    name = f"fold_{fold:02d}_seed_{seed}" if seed is not None else f"fold_{fold:02d}"
    root = OUTPUT / model
    path = root / f"{name}.json"
    pred = root / f"{name}_predictions.csv"
    if path.exists() != pred.exists():
        raise RuntimeError(f"Partial result {name}: inspect before rerunning")
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model") != model or value.get("fold") != fold or value.get("seed") != seed or value.get("test_set_used") is not False:
        raise ValueError(f"Incorrect E3 run metadata: {path}")
    return value


def report():
    """Připraví úplnou výsledkovou zprávu E3 včetně ablací E1/E2."""
    import report_e3_full
    report_e3_full.write_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Compile report after all model runs")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        create_folds()


if __name__ == "__main__":
    main()
