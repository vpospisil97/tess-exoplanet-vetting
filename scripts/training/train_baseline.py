
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "full_dataset"

INPUT_DATA = DATA / "full_cnn_views.npz"
INPUT_SPLITS = DATA / "cnn_split_indices.npz"

OUT_METRICS = DATA / "baseline_validation_metrics.csv"
OUT_PREDICTIONS = DATA / "baseline_validation_predictions.csv"
OUT_MODEL = DATA / "baseline_logistic_regression.joblib"
OUT_CONFIG = DATA / "baseline_config.json"


def load_dataset():
    if not INPUT_DATA.exists():
        raise FileNotFoundError(INPUT_DATA)

    if not INPUT_SPLITS.exists():
        raise FileNotFoundError(INPUT_SPLITS)

    with np.load(
        INPUT_DATA,
        allow_pickle=False,
    ) as data:
        local = data["x_local"].astype(np.float32)
        global_view = data["x_global"].astype(np.float32)

        y = data["y"].astype(np.int64)
        tics = data["tic_key"].astype(str)
        uids = data["signal_uid"].astype(str)

    with np.load(
        INPUT_SPLITS,
        allow_pickle=False,
    ) as data:
        train_idx = data["train_idx"].astype(np.int64)
        val_idx = data["validation_idx"].astype(np.int64)
        test_idx = data["test_idx"].astype(np.int64)

    n = len(y)

    if local.shape != (n, 101, 2):
        raise ValueError(
            f"Invalid local shape: {local.shape}"
        )

    if global_view.shape != (n, 201, 2):
        raise ValueError(
            f"Invalid global shape: {global_view.shape}"
        )

    if len(tics) != n or len(uids) != n:
        raise ValueError("Inconsistent dataset lengths")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    if not np.isfinite(local).all():
        raise ValueError("Local data contain NaN/Inf")

    if not np.isfinite(global_view).all():
        raise ValueError("Global data contain NaN/Inf")

    indices = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    if not np.array_equal(
        np.sort(indices),
        np.arange(n),
    ):
        raise ValueError(
            "Split indices are incomplete or overlapping"
        )

    tic_groups = [
        set(tics[train_idx]),
        set(tics[val_idx]),
        set(tics[test_idx]),
    ]

    if (
        tic_groups[0] & tic_groups[1]
        or tic_groups[0] & tic_groups[2]
        or tic_groups[1] & tic_groups[2]
    ):
        raise ValueError(
            "TIC leakage detected between splits"
        )

    print("TIC leakage check: OK")

    return {
        "local": local,
        "global": global_view,
        "y": y,
        "tics": tics,
        "uids": uids,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def build_features(local, global_view):
    """
    Jednoducha baseline pouziva:
    - 101 lokalnich bodu
    - 201 globalnich bodu
    - masky dostupnosti mereni

    Celkem 604 vstupnich hodnot.
    """
    n = len(local)

    local_flat = local.reshape(n, -1)
    global_flat = global_view.reshape(n, -1)

    features = np.concatenate(
        [local_flat, global_flat],
        axis=1,
    )

    return features.astype(np.float32)


def calculate_metrics(y_true, probability):
    prediction = (
        probability >= 0.5
    ).astype(np.int64)

    matrix = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )

    result = {
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }

    return result


def print_metrics(title, result):
    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)

    print(
        f"Accuracy:          {result['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: {result['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision:         {result['precision']:.4f}"
    )

    print(
        f"Recall:            {result['recall']:.4f}"
    )

    print(
        f"F1-score:          {result['f1']:.4f}"
    )

    print(
        f"Average Precision: {result['average_precision']:.4f}"
    )

    print(
        f"ROC-AUC:           {result['roc_auc']:.4f}"
    )

    print("\nConfusion matrix:")
    print("                      Predicted 0   Predicted 1")

    print(
        "Actual 0:          "
        f"{result['true_negative']:11d}"
        f"{result['false_positive']:14d}"
    )

    print(
        "Actual 1:          "
        f"{result['false_negative']:11d}"
        f"{result['true_positive']:14d}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="TESS baseline classification"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TESS - BASELINE KLASIFIKACE TRANZITNICH SIGNALU")
    print("=" * 70)

    print("\nNacitam dataset a rozdeleni...")

    data = load_dataset()

    features = build_features(
        data["local"],
        data["global"],
    )

    y = data["y"]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]

    # Test set intentionally remains untouched.
    # We only use train and validation.

    X_train = features[train_idx]
    X_val = features[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print("\nDATASET")

    print("All signals:", len(y))
    print("Train signals:", len(train_idx))
    print("Validation signals:", len(val_idx))
    print("Test signals: RESERVED")
    print("Number of features:", features.shape[1])

    print("\nTRAIN CLASS DISTRIBUTION")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nVALIDATION CLASS DISTRIBUTION")

    print(
        pd.Series(y_val)
        .value_counts()
        .sort_index()
        .to_string()
    )

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "Training dataset must contain both classes"
        )

    if len(np.unique(y_val)) != 2:
        raise ValueError(
            "Validation dataset must contain both classes"
        )

    print("\nTraining majority baseline...")

    dummy = DummyClassifier(
        strategy="prior"
    )

    dummy.fit(
        X_train,
        y_train,
    )

    dummy_probability = (
        dummy.predict_proba(X_val)[:, 1]
    )

    dummy_metrics = calculate_metrics(
        y_val,
        dummy_probability,
    )

    print("Training logistic regression...")

    baseline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=4000,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    baseline.fit(
        X_train,
        y_train,
    )

    probability = baseline.predict_proba(
        X_val
    )[:, 1]

    logistic_metrics = calculate_metrics(
        y_val,
        probability,
    )

    print_metrics(
        "MAJORITY BASELINE - VALIDATION",
        dummy_metrics,
    )

    print_metrics(
        "LOGISTIC REGRESSION - VALIDATION",
        logistic_metrics,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "majority",
                "split": "validation",
                **dummy_metrics,
            },
            {
                "model": "logistic_regression",
                "split": "validation",
                **logistic_metrics,
            },
        ]
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    predictions = pd.DataFrame(
        {
            "signal_uid": data["uids"][val_idx],
            "tic_key": data["tics"][val_idx],
            "true_label": y_val,
            "majority_probability": dummy_probability,
            "logistic_probability": probability,
            "logistic_prediction": (
                probability >= 0.5
            ).astype(int),
        }
    )

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    joblib.dump(
        {
            "model": baseline,
            "seed": args.seed,
            "feature_count": features.shape[1],
            "label_0": "false_positive",
            "label_1": "confirmed",
        },
        OUT_MODEL,
    )

    config = {
        "seed": args.seed,
        "classifier": "LogisticRegression",
        "regularization_C": 0.1,
        "solver": "liblinear",
        "class_weight": "balanced",
        "threshold": 0.5,
        "features": (
            "Flattened local and global "
            "flux and coverage masks"
        ),
        "positive_class": "confirmed",
        "negative_class": "false_positive",
        "evaluation_split": "validation",
        "test_set_used": False,
    }

    with open(
        OUT_CONFIG,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nValidation comparison:")

    print(
        metrics[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "average_precision",
                "roc_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOutput files:")
    print(OUT_METRICS)
    print(OUT_PREDICTIONS)
    print(OUT_MODEL)
    print(OUT_CONFIG)

    print("\nTest data were not evaluated.")
    print("DONE")


if __name__ == "__main__":
    main()
