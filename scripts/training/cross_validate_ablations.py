"""Doplňkové E1/E2 ablace v pěti vnějších TIC foldech a třech seedech.

Původní experimenty nejsou změněny. Obě varianty „oba pohledy“
a „s maskou“ sdílejí výsledky základní CNN z cross_validate_cnn.py.
Neaktivní větev E1 se nuluje za enkodérem; E2 nuluje druhý
vstupní kanál, nikoliv samotný flux.

Spouštění v kořeni projektu:
    python -u scripts/training/cross_validate_ablations.py --experiment all
    python -u scripts/training/cross_validate_ablations.py --experiment e1
    python -u scripts/training/cross_validate_ablations.py --experiment e2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_e3_folds as cv
import cross_validate_cnn as common
import train_cnn as original

HP = common.HP
VARIANTS = {
    "e1": (("e1_local_only", "local_only"), ("e1_global_only", "global_only")),
    "e2": (("e2_without_mask", "without_mask"),),
}


class FeatureAblationCNN(original.TessCNN):
    """Zachová parametry CNN a vynechá příznaky jedné větve až za enkodérem."""

    def __init__(self, mode: str, dropout: float):
        super().__init__(dropout=dropout)
        if mode not in {"local_only", "global_only"}:
            raise ValueError(f"Neznámá varianta E1: {mode}")
        self.mode = mode

    def forward(self, local: torch.Tensor, global_view: torch.Tensor) -> torch.Tensor:
        """Spojí pouze aktivní příznaky do původní klasifikační hlavy."""
        local_features = self.local_encoder(local)
        global_features = self.global_encoder(global_view)
        if self.mode == "local_only":
            global_features = torch.zeros_like(global_features)
        else:
            local_features = torch.zeros_like(local_features)
        return self.classifier(
            torch.cat((local_features, global_features), dim=1)
        ).squeeze(1)


def prepare_inputs(data: dict, mode: str):
    """Aplikuje stejnou transformaci toku jako hlavní CNN a případně vypne masku."""
    local = original.prepare_view(data["local"])
    global_view = original.prepare_view(data["global"])
    if mode == "without_mask":
        local = local.clone()
        global_view = global_view.clone()
        local[:, 1, :] = 0.0
        global_view[:, 1, :] = 0.0
    return local, global_view


def setup_training(data: dict, xlocal, xglobal, labels, indices, seed, mode, device):
    """Znovu inicializuje síť a připraví optimalizaci bez přístupu k hodnocenému foldu."""
    original.set_seed(seed)
    if mode in {"local_only", "global_only"}:
        model = FeatureAblationCNN(mode, HP["dropout"]).to(device)
    elif mode == "without_mask":
        model = original.TessCNN(dropout=HP["dropout"]).to(device)
    else:
        raise ValueError(f"Neznámá ablační varianta: {mode}")

    train_labels = data["y"][indices]
    positive = int(np.count_nonzero(train_labels == 1))
    negative = int(np.count_nonzero(train_labels == 0))
    if min(positive, negative) <= 0:
        raise ValueError("Vnitřní trénovací část obsahuje jen jednu třídu")

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(xlocal[indices], xglobal[indices], labels[indices]),
        batch_size=HP["batch_size"], shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    return model, loss_fn, optimizer, loader


def choose_epochs(data, xlocal, xglobal, labels, outer_train, fold, seed, mode, device):
    """Vybere počet epoch pouze na vnitřní TIC-validaci daného vnějšího tréninku."""
    inner_train, inner_val = cv.inner_group_split(data, outer_train, seed, fold)
    model, loss_fn, optimizer, loader = setup_training(
        data, xlocal, xglobal, labels, inner_train, seed, mode, device
    )
    best_ap, best_epoch, no_improvement = -1.0, 0, 0
    for epoch in range(1, HP["max_epochs"] + 1):
        common.epoch(model, loss_fn, optimizer, loader, device)
        probabilities = common.predict(model, xlocal, xglobal, inner_val, device)
        ap = cv.metrics(data["y"][inner_val], probabilities)["average_precision"]
        if ap > best_ap + 0.0001:
            best_ap, best_epoch, no_improvement = ap, epoch, 0
        else:
            no_improvement += 1
        if no_improvement >= HP["patience"]:
            break
    if best_epoch <= 0:
        raise RuntimeError("Výběr epochy selhal")
    return best_epoch, best_ap, len(inner_train), len(inner_val)


def run_variant(data, folds, xlocal, xglobal, mode, output_model, max_folds, max_seeds, device):
    """Provede vnější CV: po vnitřním výběru epoch model znovu natrénuje."""
    labels = torch.from_numpy(data["y"].astype(np.float32))
    for fold, (outer_train, outer_val) in enumerate(folds[:max_folds], start=1):
        for seed in cv.SEEDS[:max_seeds]:
            if cv.load_complete(output_model, fold, seed) is not None:
                print(f"{output_model} fold {fold}, seed {seed}: již hotovo", flush=True)
                continue
            # Stejná dvojice fold/seed musí být vypočítána také pro společnou referenční CNN.
            if cv.load_complete("cnn", fold, seed) is None:
                raise RuntimeError(
                    f"Chybí referenční CNN pro fold {fold}, seed {seed}. "
                    "Nejprve spusť cross_validate_cnn.py."
                )
            print(f"{output_model}: fold {fold}/5, seed {seed} – vnitřní výběr epoch", flush=True)
            start = time.perf_counter()
            epoch_count, inner_ap, inner_n, inner_val_n = choose_epochs(
                data, xlocal, xglobal, labels, outer_train, fold, seed, mode, device
            )
            model, loss_fn, optimizer, loader = setup_training(
                data, xlocal, xglobal, labels, outer_train, seed, mode, device
            )
            for _ in range(epoch_count):
                common.epoch(model, loss_fn, optimizer, loader, device)
            probabilities = common.predict(model, xlocal, xglobal, outer_val, device)
            result = {
                "experiment": "E1" if mode in {"local_only", "global_only"} else "E2",
                "ablation_mode": mode,
                "ablation_method": "zero_encoder_features" if mode in {"local_only", "global_only"}
                                   else "zero_explicit_coverage_channel",
                "shared_reference_model": "cnn",
                "hyperparameters": HP,
                "epoch_selected_using_inner_tic_validation": epoch_count,
                "inner_best_ap": inner_ap,
                "inner_train_signals": inner_n,
                "inner_validation_signals": inner_val_n,
                "outer_training_signals": len(outer_train),
                "training_seconds_including_inner": round(time.perf_counter() - start, 2),
                "metrics": cv.metrics(data["y"][outer_val], probabilities),
            }
            cv.save_run(output_model, fold, seed, result, outer_val, data, probabilities)
    print(f"Varianta {output_model} dokončena pro požadované foldy/seedy.", flush=True)


def main():
    """Spustí všechny nebo vybrané ablace nad již uloženými foldy E3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e1", "e2", "all"), default="all")
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--max-seeds", type=int, default=3)
    args = parser.parse_args()
    if not (1 <= args.max_folds <= 5 and 1 <= args.max_seeds <= 3):
        parser.error("--max-folds musí být 1..5 a --max-seeds musí být 1..3")
    data = cv.read_source()
    folds = cv.load_folds(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"E3 E1/E2: zařízení {device}; původní test zůstává stranou", flush=True)

    experiments = ("e1", "e2") if args.experiment == "all" else (args.experiment,)
    for experiment in experiments:
        for output_model, mode in VARIANTS[experiment]:
            xlocal, xglobal = prepare_inputs(data, mode)
            run_variant(data, folds, xlocal, xglobal, mode, output_model,
                        args.max_folds, args.max_seeds, device)
    print("E3 ablace dokončeny. Původní výsledky E1/E2 ani checkpointy se nemění.")


if __name__ == "__main__":
    main()
