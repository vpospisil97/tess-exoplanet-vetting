
from pathlib import Path
import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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


# Our run.py launcher preserves legacy root paths.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "full_dataset"

INPUT = DATA / "full_cnn_views.npz"
SPLITS = DATA / "cnn_split_indices.npz"

OUTPUT = DATA / "cnn_model_a"

LOCAL_LENGTH = 101
GLOBAL_LENGTH = 201


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not SPLITS.exists():
        raise FileNotFoundError(SPLITS)

    with np.load(INPUT, allow_pickle=False) as z:
        local = z["x_local"].astype(np.float32)
        global_view = z["x_global"].astype(np.float32)
        labels = z["y"].astype(np.int64)

        tics = z["tic_key"].astype(str)
        uids = z["signal_uid"].astype(str)

    with np.load(SPLITS, allow_pickle=False) as z:
        train_idx = z["train_idx"].astype(np.int64)
        val_idx = z["validation_idx"].astype(np.int64)
        test_idx = z["test_idx"].astype(np.int64)

    n = len(labels)

    if local.shape != (n, LOCAL_LENGTH, 2):
        raise ValueError(
            f"Wrong local shape: {local.shape}"
        )

    if global_view.shape != (n, GLOBAL_LENGTH, 2):
        raise ValueError(
            f"Wrong global shape: {global_view.shape}"
        )

    if len(tics) != n or len(uids) != n:
        raise ValueError("Inconsistent sample counts")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    if not np.isfinite(local).all():
        raise ValueError("Invalid local data")

    if not np.isfinite(global_view).all():
        raise ValueError("Invalid global data")

    all_indices = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    if not np.array_equal(
        np.sort(all_indices),
        np.arange(n),
    ):
        raise ValueError(
            "Overlapping or incomplete split indices"
        )

    groups = [
        set(tics[train_idx]),
        set(tics[val_idx]),
        set(tics[test_idx]),
    ]

    if (
        groups[0] & groups[1]
        or groups[0] & groups[2]
        or groups[1] & groups[2]
    ):
        raise ValueError(
            "TIC data leakage detected"
        )

    print("TIC leakage check: OK")

    return {
        "local": local,
        "global": global_view,
        "y": labels,
        "tics": tics,
        "uids": uids,
        "train": train_idx,
        "validation": val_idx,
    }


def prepare_view(view):
    """
    Input format:
      channel 0 = relative flux
      channel 1 = coverage mask

    Convert flux to parts per thousand (ppt)
    and apply a fixed smooth transformation.

    No statistics from validation/test are used.
    """
    flux = view[:, :, 0]
    mask = view[:, :, 1]

    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError("Invalid coverage mask")

    flux_ppt = flux * 1000.0

    # Robust scaling preserving sign and depth.
    transformed = np.arcsinh(
        flux_ppt / 5.0
    )

    transformed = np.where(
        mask > 0,
        transformed,
        0.0,
    )

    combined = np.stack(
        [transformed, mask],
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(combined).all():
        raise ValueError(
            "Transformed data contain invalid values"
        )

    return torch.from_numpy(
        np.ascontiguousarray(combined)
    )


class ConvEncoder(nn.Module):
    """
    One-dimensional CNN encoder for one view.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                2, 32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64, 64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.features(x)

        average = self.avg_pool(x).squeeze(-1)
        maximum = self.max_pool(x).squeeze(-1)

        return torch.cat(
            [average, maximum],
            dim=1,
        )


class TessCNN(nn.Module):
    """
    Model A: two-branch 1D CNN.
    Trained from random initialization.
    """

    def __init__(self, dropout=0.30):
        super().__init__()

        self.local_encoder = ConvEncoder()
        self.global_encoder = ConvEncoder()

        # Each encoder produces 128 features.
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, local, global_view):
        a = self.local_encoder(local)
        b = self.global_encoder(global_view)

        combined = torch.cat(
            [a, b],
            dim=1,
        )

        return self.classifier(
            combined
        ).squeeze(1)


def metrics_from_probabilities(y, p):
    predicted = (
        p >= 0.5
    ).astype(np.int64)

    cm = confusion_matrix(
        y,
        predicted,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(y, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted)
        ),
        "precision": float(
            precision_score(
                y, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, predicted, zero_division=0
            )
        ),
        "average_precision": float(
            average_precision_score(y, p)
        ),
        "roc_auc": float(
            roc_auc_score(y, p)
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    all_y = []
    all_p = []

    total_loss = 0.0
    total_samples = 0

    for local, global_view, y in loader:
        local = local.to(device)
        global_view = global_view.to(device)
        y = y.to(device)

        logits = model(
            local,
            global_view,
        )

        loss = loss_fn(logits, y)

        batch_size = len(y)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        probabilities = torch.sigmoid(
            logits
        )

        all_y.append(
            y.cpu().numpy()
        )

        all_p.append(
            probabilities.cpu().numpy()
        )

    y_true = np.concatenate(
        all_y
    ).astype(np.int64)

    probabilities = np.concatenate(
        all_p
    )

    result = metrics_from_probabilities(
        y_true,
        probabilities,
    )

    result["loss"] = (
        total_loss / total_samples
    )

    return result, y_true, probabilities


def save_curves(history, path):
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4),
    )

    axes[0].plot(
        df["epoch"],
        df["train_loss"],
        label="Train",
    )

    axes[0].plot(
        df["epoch"],
        df["val_loss"],
        label="Validation",
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        df["epoch"],
        df["val_ap"],
        label="CNN validation AP",
    )

    axes[1].axhline(
        0.747872,
        linestyle="--",
        label="Logistic regression AP",
    )

    axes[1].set_title("Average Precision")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=170,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TESS Model A - two-view 1D CNN"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.patience < 1:
        parser.error("--patience must be >= 1")

    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0,1)")

    print("=" * 72)
    print("TESS - MODEL A - 1D CNN")
    print("=" * 72)

    set_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available"
            )
        device = torch.device("cuda")

    elif args.device == "cpu":
        device = torch.device("cpu")

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("\nPyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nLoading dataset...")

    data = load_data()

    x_local = prepare_view(
        data["local"]
    )

    x_global = prepare_view(
        data["global"]
    )

    y = torch.from_numpy(
        data["y"].astype(np.float32)
    )

    train_idx = data["train"]
    val_idx = data["validation"]

    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test: reserved, not evaluated")

    print("Local tensor:", tuple(x_local.shape))
    print("Global tensor:", tuple(x_global.shape))

    train_y = data["y"][train_idx]

    negatives = int(
        np.sum(train_y == 0)
    )

    positives = int(
        np.sum(train_y == 1)
    )

    print("\nTrain class distribution:")
    print("False positives:", negatives)
    print("Confirmed:", positives)

    if negatives == 0 or positives == 0:
        raise ValueError(
            "Both training classes are required"
        )

    train_dataset = TensorDataset(
        x_local[train_idx],
        x_global[train_idx],
        y[train_idx],
    )

    val_dataset = TensorDataset(
        x_local[val_idx],
        x_global[val_idx],
        y[val_idx],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TessCNN(
        dropout=args.dropout
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", parameter_count)

    # Positive class = confirmed (1).
    # Since it is the majority class, its
    # loss weight is less than 1.
    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = f"seed_{args.seed}"

    checkpoint_path = OUTPUT / f"cnn_best_{prefix}.pt"
    metrics_path = OUTPUT / f"metrics_{prefix}.json"
    history_path = OUTPUT / f"history_{prefix}.csv"
    predictions_path = OUTPUT / f"predictions_{prefix}.csv"
    figure_path = OUTPUT / f"curves_{prefix}.png"

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = []

    start_time = time.perf_counter()

    print("\nTraining started")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for local, global_view, labels in train_loader:
            local = local.to(
                device,
                non_blocking=True,
            )

            global_view = global_view.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                local,
                global_view,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = len(labels)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss / total_samples
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        val_ap = val_metrics[
            "average_precision"
        ]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ap": val_ap,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_balanced_accuracy": (
                val_metrics["balanced_accuracy"]
            ),
            "val_f1": val_metrics["f1"],
        })

        improved = (
            val_ap > best_ap + 0.0001
        )

        if improved:
            best_ap = val_ap
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "seed": args.seed,
                    "epoch": best_epoch,
                    "best_val_ap": best_ap,
                    "parameters": parameter_count,
                    "dropout": args.dropout,
                    "transform": "asinh(flux_ppt/5)",
                },
                checkpoint_path,
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_AP={val_ap:.4f} | "
            f"val_ROC={val_metrics['roc_auc']:.4f} | "
            f"val_BAcc={val_metrics['balanced_accuracy']:.4f}"
            + (" | BEST" if improved else ""),
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print(
                "\nEarly stopping triggered."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint was produced"
        )

    # Restore the checkpoint with the
    # best validation Average Precision.
    model.load_state_dict(best_state)

    final_metrics, true_y, probabilities = evaluate(
        model,
        val_loader,
        loss_fn,
        device,
    )

    elapsed = time.perf_counter() - start_time

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    predictions = pd.DataFrame({
        "signal_uid": data["uids"][val_idx],
        "tic_key": data["tics"][val_idx],
        "true_label": true_y,
        "predicted_probability": probabilities,
        "predicted_label": (
            probabilities >= 0.5
        ).astype(int),
    })

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    save_curves(
        history,
        figure_path,
    )

    result = {
        "model": "two_view_1d_cnn",
        "seed": args.seed,
        "device": str(device),
        "train_signals": len(train_idx),
        "validation_signals": len(val_idx),
        "test_set_used": False,
        "trainable_parameters": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "training_seconds": round(elapsed, 2),
        "hyperparameters": vars(args),
        "validation": final_metrics,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("SUMMARY - BEST CNN CHECKPOINT")
    print("=" * 72)

    print("Best epoch:", best_epoch)
    print("Epochs completed:", len(history))
    print("Training seconds:", round(elapsed, 2))
    print("Parameters:", parameter_count)

    print("\nVALIDATION METRICS")

    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
    ):
        print(
            f"{key:20s}: "
            f"{final_metrics[key]:.4f}"
        )

    print("\nCONFUSION MATRIX")
    print(
        "TN:", final_metrics["tn"],
        "| FP:", final_metrics["fp"],
    )

    print(
        "FN:", final_metrics["fn"],
        "| TP:", final_metrics["tp"],
    )

    print("\nBASELINE REFERENCE")
    print("Logistic regression validation AP: 0.7479")
    print("Logistic regression validation ROC-AUC: 0.6415")

    print("\nOUTPUT FILES")
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)
    print("History:", history_path)
    print("Predictions:", predictions_path)
    print("Curves:", figure_path)

    print("\nTEST SET NOT USED")
    print("DONE")


if __name__ == "__main__":
    main()
