"""E3: five TIC-grouped outer folds x three CNN seeds; nested epoch selection.

Run from repository root:
  python -u scripts/training/cross_validate_cnn.py --max-folds 1 --max-seeds 1
  python -u scripts/training/cross_validate_cnn.py
The original held-out test is never evaluated.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_e3_folds as cv
import train_cnn as original

HP = {'learning_rate': 0.001, 'weight_decay': 0.0001, 'dropout': 0.3,
      'batch_size': 32, 'max_epochs': 35, 'patience': 7, 'threshold': 0.5}


def make_training(data, xlocal, xglobal, y_tensor, train_idx, seed, device):
    original.set_seed(seed)
    model = original.TessCNN(dropout=HP['dropout']).to(device)
    neg = int((data['y'][train_idx] == 0).sum())
    pos = int((data['y'][train_idx] == 1).sum())
    if min(neg, pos) == 0:
        raise ValueError('CNN train fold contains only one class')
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=HP['learning_rate'],
                              weight_decay=HP['weight_decay'])
    loader = DataLoader(TensorDataset(xlocal[train_idx], xglobal[train_idx], y_tensor[train_idx]),
                        batch_size=HP['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    return model, loss, optim, loader


def epoch(model, loss_fn, optimizer, loader, device):
    model.train()
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(a, b), y)
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def predict(model, xlocal, xglobal, indices, device):
    model.eval()
    results = []
    for start in range(0, len(indices), 128):
        idx = indices[start:start + 128]
        p = torch.sigmoid(model(xlocal[idx].to(device), xglobal[idx].to(device)))
        results.append(p.cpu().numpy())
    return np.concatenate(results).astype(np.float64)


def select_epoch(data, xlocal, xglobal, y_tensor, outer_train, fold, seed, device):
    inner_train, inner_val = cv.inner_group_split(data, outer_train, seed, fold)
    model, loss, optim, loader = make_training(data, xlocal, xglobal, y_tensor,
                                               inner_train, seed, device)
    best_ap, best_epoch, waiting = -1.0, 0, 0
    for ep in range(1, HP['max_epochs'] + 1):
        epoch(model, loss, optim, loader, device)
        p = predict(model, xlocal, xglobal, inner_val, device)
        ap = cv.metrics(data['y'][inner_val], p)['average_precision']
        if ap > best_ap + 0.0001:
            best_ap, best_epoch, waiting = ap, ep, 0
        else:
            waiting += 1
        if waiting >= HP['patience']:
            break
    return best_epoch, best_ap, len(inner_train), len(inner_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--max-seeds', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5 or not 1 <= args.max_seeds <= 3:
        parser.error('max-folds 1..5 and max-seeds 1..3 are required')
    data = cv.read_source()
    folds = cv.load_folds(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('E3 CNN device:', device)
    xlocal = original.prepare_view(data['local'])
    xglobal = original.prepare_view(data['global'])
    y_tensor = torch.from_numpy(data['y'].astype(np.float32))
    for fold, (outer_train, outer_val) in enumerate(folds[:args.max_folds], 1):
        for seed in cv.SEEDS[:args.max_seeds]:
            if cv.load_complete('cnn', fold, seed) is not None:
                print(f'Fold {fold}, seed {seed}: already completed')
                continue
            print(f'CNN E3 fold {fold}/5 seed {seed} – inner epoch selection', flush=True)
            start = time.perf_counter()
            best_epoch, inner_ap, inner_n, inner_val_n = select_epoch(
                data, xlocal, xglobal, y_tensor, outer_train, fold, seed, device)
            print(f'Inner best epoch {best_epoch}; AP {inner_ap:.4f}; '
                  f'inner train/val {inner_n}/{inner_val_n}', flush=True)
            model, loss, optimizer, loader = make_training(data, xlocal, xglobal,
                                                            y_tensor, outer_train, seed, device)
            for _ in range(best_epoch):
                epoch(model, loss, optimizer, loader, device)
            p = predict(model, xlocal, xglobal, outer_val, device)
            result = {'hyperparameters': HP, 'epoch_selected_using_inner_tic_validation': best_epoch,
                      'inner_best_ap': inner_ap, 'inner_train_signals': inner_n,
                      'inner_validation_signals': inner_val_n,
                      'outer_training_signals': len(outer_train),
                      'training_seconds_including_inner': round(time.perf_counter() - start, 2),
                      'metrics': cv.metrics(data['y'][outer_val], p)}
            cv.save_run('cnn', fold, seed, result, outer_val, data, p)
    print('CNN E3 finished for requested folds/seeds. Original checkpoints untouched.')

if __name__ == '__main__':
    main()
