
"""
TESS exoplanet vetting: frozen evaluation protocol.

Run from the repository root:
    python -u scripts/evaluation/evaluate_tess_final.py --preflight
    python -u scripts/evaluation/evaluate_tess_final.py --test

PRE-FLIGHT uses VALIDATION only. Reproduces saved validation probabilities
for the logistic baseline, three tuned CNN checkpoints, and three tuned
Mantis heads. It also checks 16 validation embeddings against a fresh pass
through the pretrained Mantis encoder.

TEST is blocked until preflight succeeds. It evaluates the already selected
models once on the held-out TIC-disjoint 290 test signals. A lock file is
created before testing to guard against accidental reruns/test tuning.

Checkpoint files and joblib are deserialized: use only files you trained
and trust. Mantis may load the original pretrained weights from Hugging Face.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "full_dataset"
VIEWS = DATA / "full_cnn_views.npz"
SPLITS = DATA / "cnn_split_indices.npz"
MANTIS_DIR = DATA / "mantis_model_b"
CNN_DIR = DATA / "cnn_tuned_repeats" / "lr_0p001__drop_0p3__batch_32"
HEAD_DIR = MANTIS_DIR / "head_grid_v1" / "hidden_128__drop_0p5"
SELECTION = MANTIS_DIR / "head_grid_v1" / "best_head_config.json"
BASELINE = DATA / "baseline_logistic_regression.joblib"
VALIDATION_FEATURES = MANTIS_DIR / "features_validation.npz"
TRAIN_FEATURES = MANTIS_DIR / "features_train.npz"
OUT = DATA / "final_test"
PREFLIGHT = OUT / "preflight_pass.json"
LOCK = OUT / "test_started.json"
REPORT = ROOT / "docs" / "final_test_results.md"
SEEDS = (42, 123, 2026)
THRESHOLD = 0.5
EXPECTED_HASHES = {
    "views": "ec8b3ef69c3f591b0aa027309ccdbef6e85c9943a838577d031e1c15461f8fb8",
    "splits": "d53c2e68ad021d4a4dd6915da702e488129f92af1a654c5ba007de2723199c54",
}
SCRIPTS = ROOT / "scripts" / "training"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_file(path):
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, filename):
    path = SCRIPTS / filename
    require(path.is_file(), f"Missing trainer source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_dataset():
    for key, path in (("views", VIEWS), ("splits", SPLITS)):
        actual = hash_file(path)
        require(actual == EXPECTED_HASHES[key],
                f"Dataset fingerprint changed: {path}")
    with np.load(VIEWS, allow_pickle=False) as z:
        local = z["x_local"].astype(np.float32)
        glob = z["x_global"].astype(np.float32)
        y = z["y"].astype(np.int64)
        uid = z["signal_uid"].astype(str)
        tic = z["tic_key"].astype(str)
    with np.load(SPLITS, allow_pickle=False) as z:
        idx = {name: z[f"{name}_idx"].astype(np.int64)
               for name in ("train", "validation", "test")}
    require(local.shape == (1918, 101, 2), "Wrong local view shape")
    require(glob.shape == (1918, 201, 2), "Wrong global view shape")
    require(y.shape == uid.shape == tic.shape == (1918,), "Wrong metadata shape")
    require(np.isfinite(local).all() and np.isfinite(glob).all(),
            "Non-finite input views")
    require(set(np.unique(y)) == {0, 1}, "Wrong class labels")
    require(len(set(uid)) == 1918, "Duplicate signal identifiers")
    require([len(idx[k]) for k in idx] == [1335, 293, 290],
            "Unexpected split sizes")
    together = np.concatenate(list(idx.values()))
    require(np.array_equal(np.sort(together), np.arange(1918)),
            "Overlapping or incomplete split indices")
    groups = {k: set(tic[v].tolist()) for k, v in idx.items()}
    require(not (groups["train"] & groups["validation"] or
                 groups["train"] & groups["test"] or
                 groups["validation"] & groups["test"]),
            "TIC leakage across splits")
    print("Dataset SHA256: OK; 1918 signals; TIC leakage: NONE")
    return {"local": local, "global": glob, "y": y, "uid": uid,
            "tic": tic, "idx": idx}


def selected_head_config():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    config = selection["config"]
    require(selection["selected_trial"] == "hidden_128__drop_0p5" and
            config["hidden_size"] == 128 and config["dropout"] == 0.5 and
            config["threshold"] == THRESHOLD and
            selection["test_set_used"] is False,
            "Unexpected Mantis selection config")
    for key, path in (("train", TRAIN_FEATURES),
                      ("validation", VALIDATION_FEATURES)):
        require(hash_file(path) == config["feature_hashes"][key],
                f"Mantis {key} embedding source differs from selected grid")
    return config


def checkpoint_files():
    files = [VIEWS, SPLITS, BASELINE, SELECTION, TRAIN_FEATURES,
             VALIDATION_FEATURES]
    files += [SCRIPTS / name for name in
              ("train_cnn.py", "train_baseline.py",
               "extract_mantis_features.py", "train_mantis_head.py")]
    files += [CNN_DIR / f"cnn_best_seed_{seed}.pt" for seed in SEEDS]
    files += [HEAD_DIR / f"seed_{seed}" / "best_head.pt" for seed in SEEDS]
    return {str(path.relative_to(ROOT)): hash_file(path) for path in files}


def read_validation_csv(path, data, probabilities, column):
    require(path.is_file(), f"Missing validation predictions: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    idx = data["idx"]["validation"]
    require(len(rows) == len(idx), f"Unexpected row count in {path}")
    lookup = {row["signal_uid"]: row for row in rows}
    require(len(lookup) == len(idx), f"Duplicate validation UIDs: {path}")
    reference = []
    for i in idx:
        uid = data["uid"][i]
        require(uid in lookup, f"Missing validation UID {uid}: {path}")
        row = lookup[uid]
        require(str(row["tic_key"]) == data["tic"][i],
                f"TIC mismatch: {uid}")
        require(int(row["true_label"]) == int(data["y"][i]),
                f"Label mismatch: {uid}")
        reference.append(float(row[column]))
    reference = np.asarray(reference)
    difference = float(np.max(np.abs(reference - probabilities)))
    require(np.allclose(reference, probabilities, rtol=0.001, atol=0.001),
            f"Validation reproduction failed for {path}: max diff={difference}")
    return difference


def scores(y, prob):
    require(len(y) == len(prob) and np.isfinite(prob).all(),
            "Invalid model probabilities")
    require(np.all((prob >= 0) & (prob <= 1)), "Scores outside [0, 1]")
    binary = (prob >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, binary)),
        "f1": float(f1_score(y, binary, zero_division=0)),
        "accuracy": float(accuracy_score(y, binary)),
        "precision": float(precision_score(y, binary, zero_division=0)),
        "recall": float(recall_score(y, binary, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_baseline(data, split, baseline_module):
    stored = joblib.load(BASELINE)  # User's own trained, trusted file only.
    require(stored.get("feature_count") == 604 and
            stored.get("label_1") == "confirmed", "Wrong baseline metadata")
    idx = data["idx"][split]
    x = baseline_module.build_features(data["local"][idx],
                                       data["global"][idx])
    require(x.shape == (len(idx), 604), "Wrong baseline feature shape")
    require(np.array_equal(stored["model"].classes_, [0, 1]),
            "Unexpected baseline class ordering")
    start = time.perf_counter()
    p = stored["model"].predict_proba(x)[:, 1]
    seconds = time.perf_counter() - start
    if split == "validation":
        delta = read_validation_csv(
            DATA / "baseline_validation_predictions.csv", data, p,
            "logistic_probability")
        print(f"Logistic baseline validation reproduction: PASS (max diff {delta:.2g})")
    return p, seconds


def predict_cnn(data, split, cnn, device):
    idx = data["idx"][split]
    local = cnn.prepare_view(data["local"][idx])
    glob = cnn.prepare_view(data["global"][idx])
    results = {}
    for seed in SEEDS:
        path = CNN_DIR / f"cnn_best_seed_{seed}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(saved["seed"] == seed and saved["dropout"] == 0.3 and
                saved["transform"] == "asinh(flux_ppt/5)",
                f"CNN checkpoint config mismatch: seed {seed}")
        model = cnn.TessCNN(dropout=0.3).to(device)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.eval()
        pieces = []
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for first in range(0, len(idx), 64):
                last = first + 64
                logits = model(local[first:last].to(device),
                               glob[first:last].to(device))
                pieces.append(torch.sigmoid(logits).cpu().numpy())
        synchronize(device)
        elapsed = time.perf_counter() - start
        prob = np.concatenate(pieces)
        if split == "validation":
            delta = read_validation_csv(
                CNN_DIR / f"predictions_seed_{seed}.csv", data, prob,
                "predicted_probability")
            print(f"CNN seed {seed} validation: PASS (max diff {delta:.2g})")
        results[seed] = (prob, elapsed)
    return results


def load_head(seed, head, config, device):
    path = HEAD_DIR / f"seed_{seed}" / "best_head.pt"
    # Saved scaler_mean/scaler_scale are NumPy arrays. The original trusted
    # project checkpoint therefore requires weights_only=False.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(saved["seed"] == seed and saved["config"] == config,
            f"Mantis head configuration mismatch: seed {seed}")
    mean = np.asarray(saved["scaler_mean"], dtype=np.float32)
    scale = np.asarray(saved["scaler_scale"], dtype=np.float32)
    require(mean.shape == (1024,) and scale.shape == (1024,) and
            np.isfinite(mean).all() and np.isfinite(scale).all() and
            (scale > 0).all(), f"Invalid scaler: seed {seed}")
    head.CONFIG = config
    model = head.make_model().to(device)
    model.load_state_dict(saved["head_state_dict"], strict=True)
    model.eval()
    return model, mean, scale


def predict_heads(features, split, data, head, config, device):
    require(features.shape == (len(data["idx"][split]), 1024) and
            np.isfinite(features).all(), f"Invalid {split} Mantis features")
    results = {}
    for seed in SEEDS:
        model, mean, scale = load_head(seed, head, config, device)
        x = torch.from_numpy(((features - mean) / scale).astype(np.float32))
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            batches = []
            for first in range(0, len(x), 64):
                logits = model(x[first:first + 64].to(device)).squeeze(-1)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        synchronize(device)
        elapsed = time.perf_counter() - start
        p = np.concatenate(batches)
        if split == "validation":
            delta = read_validation_csv(
                HEAD_DIR / f"seed_{seed}" / "predictions.csv",
                data, p, "predicted_probability")
            print(f"Mantis head seed {seed} validation: PASS (max diff {delta:.2g})")
        results[seed] = (p, elapsed)
    return results


def mantis_features(data, split, extractor, trainer):
    idx = data["idx"][split]
    if split == "validation":
        idx = idx[:16]
    local = extractor.prepare_view(data["local"][idx])
    glob = extractor.prepare_view(data["global"][idx])
    start = time.perf_counter()
    a = np.asarray(trainer.transform(local, batch_size=16), dtype=np.float32)
    b = np.asarray(trainer.transform(glob, batch_size=16), dtype=np.float32)
    elapsed = time.perf_counter() - start
    features = np.concatenate([a, b], axis=1)
    require(features.shape == (len(idx), 1024) and
            np.isfinite(features).all(), f"Invalid Mantis {split} embeddings")
    if split == "validation":
        with np.load(VALIDATION_FEATURES, allow_pickle=False) as z:
            saved_idx = z["indices"][:len(idx)]
            saved_features = z["features"][:len(idx)]
        require(np.array_equal(saved_idx, idx),
                "Validation Mantis feature alignment differs")
        delta = float(np.max(np.abs(saved_features - features)))
        require(np.allclose(saved_features, features, rtol=0.001, atol=0.001),
                f"Pretrained Mantis changed: validation embedding diff={delta}")
        print(f"Mantis encoder 16 validation examples: PASS (max diff {delta:.3g})")
    return features, elapsed


def load_mantis(extractor, device):
    # Exact pretrained encoder settings from extract_mantis_features.py.
    extractor.DEVICE = device.type
    require(extractor.CHECKPOINT == "paris-noah/Mantis-8M" and
            extractor.LENGTH == 512, "Mantis source configuration changed")
    print("Loading pretrained Mantis-8M (may use its local download cache)...")
    trainer, params = extractor.load_mantis()
    print(f"Mantis encoder parameters: {params:,}")
    return trainer


def preflight(data, cnn, base, head, extractor, device, config):
    require(not LOCK.exists(), "Test was already started; no new preflight")
    print("\nPRE-FLIGHT: VALIDATION ONLY; no test predictions")
    fingerprints = checkpoint_files()
    p_base, _ = predict_baseline(data, "validation", base)
    cnn_outputs = predict_cnn(data, "validation", cnn, device)
    with np.load(VALIDATION_FEATURES, allow_pickle=False) as z:
        features = z["features"].astype(np.float32)
        require(np.array_equal(z["indices"], data["idx"]["validation"]),
                "Incorrect validation Mantis feature indices")
        idx = data["idx"]["validation"]
        require(np.array_equal(z["y"], data["y"][idx]),
                "Validation embedding labels differ")
    head_outputs = predict_heads(features, "validation", data, head, config, device)
    trainer = load_mantis(extractor, device)
    _, _ = mantis_features(data, "validation", extractor, trainer)
    print("Validation Average Precision:")
    y = data["y"][data["idx"]["validation"]]
    print(f"  Baseline: {scores(y, p_base)['average_precision']:.4f}")
    for name, outputs in (("CNN", cnn_outputs), ("Mantis", head_outputs)):
        for seed, (p, _) in outputs.items():
            print(f"  {name} seed {seed}: {scores(y, p)['average_precision']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "tess_final_test_v1", "validation_reproduction": "PASS",
        "test_used": False, "sha256": fingerprints,
        "seeds": list(SEEDS), "selected_cnn": CNN_DIR.name,
        "selected_mantis_head": HEAD_DIR.name,
    }
    PREFLIGHT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPRE-FLIGHT PASSED. Manifest: {PREFLIGHT}")
    print("TEST SET NOT EVALUATED. Send this output for review before --test.")


def summarize(results, data, timings, device):
    y = data["y"][data["idx"]["test"]]
    records = []
    for name, run in results.items():
        for seed, p in run.items():
            row = {"model": name, "seed": seed, **scores(y, p)}
            records.append(row)
            print(f"{name:10s} seed={str(seed):>4} "
                  f"AP={row['average_precision']:.4f} "
                  f"ROC={row['roc_auc']:.4f} F1={row['f1']:.4f} "
                  f"TN/FP/FN/TP={row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}")
    with (OUT / "metrics_by_model_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    idx = data["idx"]["test"]
    with (OUT / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["signal_uid", "tic_key", "true_label"] + [
            f"{name}_seed_{seed}" for name, run in results.items() for seed in run]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for j, i in enumerate(idx):
            row = {"signal_uid": data["uid"][i], "tic_key": data["tic"][i],
                   "true_label": int(y[j])}
            row.update({f"{name}_seed_{seed}": float(p[j])
                        for name, run in results.items() for seed, p in run.items()})
            writer.writerow(row)
    keys = ("average_precision", "roc_auc", "balanced_accuracy", "f1")
    summary = ["# TESS held-out test: final frozen models", "",
               "Exactly one held-out test pass, TIC-disjoint from training/validation.",
               "Selected with validation AP; positive class = confirmed; threshold = 0.5.",
               "", "## Performance (mean ± sample SD over three seeds)", "",
               "| Model | AP | ROC-AUC | Balanced accuracy | F1 |",
               "|---|---:|---:|---:|---:|"]
    for name in results:
        selected = [r for r in records if r["model"] == name]
        cells = []
        for metric in keys:
            values = np.asarray([r[metric] for r in selected])
            cells.append(f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
                         if len(values) > 1 else f"{values[0]:.4f}")
        summary.append(f"| {name} | " + " | ".join(cells) + " |")
    summary += ["", "## Per-run results", "",
                "| Model | Seed | AP | ROC-AUC | Balanced acc. | F1 | TN | FP | FN | TP |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        summary.append(f"| {row['model']} | {row['seed']} | " + " | ".join(
            f"{row[m]:.4f}" for m in keys) +
            f" | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |")
    summary += ["", "## Runtime", "",
                "Model load time excluded. Mantis encoder extraction and classifier-head"
                " inference are measured separately. CNN timing includes device transfers.",
                ""]
    for key, seconds in timings.items():
        summary.append(f"- {key}: {seconds:.3f} s")
    summary += ["", "## Protocol and limitations", "",
                f"- Test signals: {len(y)}, confirmed: {int(y.sum())}, false positives: {int((y == 0).sum())}",
                f"- Device: {device}; torch: {torch.__version__}; numpy: {np.__version__}",
                "- Checkpoints and preprocessing were frozen after validation.",
                "- No hyperparameters were chosen from test performance.",
                "- Seeds share one fixed TIC-grouped split; SD reflects initialization only.",
                "- A single test split provides limited precision for generalization claims.",
                "- Mantis embeddings were generated with the frozen pretrained encoder.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(summary).rstrip() + "\n", encoding="utf-8")
    (OUT / "test_summary.json").write_text(json.dumps({
        "test_size": int(len(y)), "results": records, "timings_seconds": timings,
        "device": str(device), "protocol": "tess_final_test_v1",
        "evaluation_complete": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nFinal report: {REPORT}")
    print("TEST EVALUATION COMPLETE — do not retune on these results")


def final_test(data, cnn, base, head, extractor, device, config):
    require(PREFLIGHT.is_file(), "Run --preflight successfully first")
    require(not LOCK.exists(), "Test already started. Do not rerun automatically")
    require(not REPORT.exists(), "Final report already exists — do not rerun")
    manifest = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    require(manifest["validation_reproduction"] == "PASS" and
            manifest["test_used"] is False and manifest["sha256"] == checkpoint_files(),
            "Data or checkpoints changed since preflight")
    # Load the encoder before locking; no held-out examples used yet.
    trainer = load_mantis(extractor, device)
    OUT.mkdir(parents=True, exist_ok=True)
    lock_payload = json.dumps({"protocol": "tess_final_test_v1",
                               "status": "TEST_STARTED", "note": "Do not tune using this split"},
                              indent=2) + "\n"
    fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(lock_payload)
    print("\nTEST RUN LOCK CREATED: first and only held-out evaluation")
    outputs = {}
    timings = {}
    base_p, duration = predict_baseline(data, "test", base)
    outputs["Logistic"] = {"baseline": base_p}
    timings["Logistic prediction"] = duration
    cnn_results = predict_cnn(data, "test", cnn, device)
    outputs["CNN"] = {seed: p for seed, (p, _) in cnn_results.items()}
    for seed, (_, duration) in cnn_results.items():
        timings[f"CNN seed {seed}"] = duration
    features, elapsed = mantis_features(data, "test", extractor, trainer)
    timings["Mantis encoder extraction, 2 views"] = elapsed
    head_results = predict_heads(features, "test", data, head, config, device)
    outputs["Mantis"] = {seed: p for seed, (p, _) in head_results.items()}
    for seed, (_, duration) in head_results.items():
        timings[f"Mantis head seed {seed}"] = duration
    summarize(outputs, data, timings, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    require(args.preflight != args.test,
            "Specify exactly one of --preflight or --test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TESS frozen evaluation | mode:", "PREFLIGHT" if args.preflight else "TEST")
    print("Device:", device)
    data = audit_dataset()
    config = selected_head_config()
    cnn = load_module("_tess_eval_cnn", "train_cnn.py")
    base = load_module("_tess_eval_baseline", "train_baseline.py")
    head = load_module("_tess_eval_head", "train_mantis_head.py")
    extractor = load_module("_tess_eval_extractor", "extract_mantis_features.py")
    if args.preflight:
        preflight(data, cnn, base, head, extractor, device, config)
    else:
        final_test(data, cnn, base, head, extractor, device, config)


if __name__ == "__main__":
    main()
