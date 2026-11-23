"""E3: five TIC-grouped outer folds x three seeds of the frozen Mantis MLP head.

Uses existing Mantis train + validation embeddings. Frozen encoder is not retrained.
Recomputes embedding scaler inside every inner/outer training partition.
The separate held-out test is not used.

Run from repository root:
  python -u scripts/training/cross_validate_mantis.py --max-folds 1 --max-seeds 1
  python -u scripts/training/cross_validate_mantis.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_e3_folds as cv
import train_cnn as original

EMBEDDINGS = cv.DATA / 'mantis_model_b'
HP = {'embedding_source': 'paris-noah/Mantis-8M frozen features',
      'hidden_size': 128, 'dropout': 0.5, 'learning_rate': 0.001,
      'weight_decay': 0.001, 'batch_size': 64, 'max_epochs': 80,
      'patience': 12, 'threshold': 0.5}


def load_embeddings(data):
    n = len(data['y'])
    features = np.full((n, 1024), np.nan, dtype=np.float32)
    observed = set()
    for split, expected in (('train', data['train']), ('validation', data['validation'])):
        path = EMBEDDINGS / f'features_{split}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing precomputed frozen Mantis features: {path}')
        with np.load(path, allow_pickle=False) as z:
            required = {'indices', 'features', 'y', 'tic_key', 'signal_uid'}
            if not required.issubset(z.files):
                raise ValueError(f'Missing embedding data in {path}')
            indices = np.asarray(z['indices'], dtype=np.int64)
            x = np.asarray(z['features'], dtype=np.float32)
            y = np.asarray(z['y'], dtype=np.int64)
            tic = np.char.strip(np.asarray(z['tic_key']).astype(str))
            uid = np.char.strip(np.asarray(z['signal_uid']).astype(str))
        if not np.array_equal(indices, expected):
            raise ValueError(f'Embedding indices differ from original {split} split')
        if x.shape != (len(indices), 1024) or not np.isfinite(x).all():
            raise ValueError(f'Invalid {split} embedding shape or values')
        if (not np.array_equal(y, data['y'][indices]) or
            not np.array_equal(tic, data['tic'][indices]) or
            not np.array_equal(uid, data['uid'][indices])):
            raise ValueError(f'Embedding metadata mismatch: {split}')
        for i in indices:
            if int(i) in observed:
                raise ValueError('Duplicate embedding index')
            observed.add(int(i))
        features[indices] = x
    if observed != set(data['dev'].tolist()):
        raise ValueError('Incomplete development embedding coverage')
    if not np.isfinite(features[data['dev']]).all():
        raise ValueError('Nonfinite development embeddings')
    # Test rows are never read from feature files or passed to a scaler/model.
    return features


def make_model(seed, device):
    original.set_seed(seed)
    return nn.Sequential(nn.Linear(1024, HP['hidden_size']), nn.ReLU(),
                         nn.Dropout(HP['dropout']), nn.Linear(HP['hidden_size'], 1)).to(device)


def epoch(model, loss_fn, optimizer, loader, device):
    model.train()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_x).squeeze(1), batch_y)
        loss.backward()
        optimizer.step()


def setup_training(data, embeddings, train_indices, seed, device):
    # No statistics from inner validation or outer validation may enter scaler.fit.
    scaler = StandardScaler()
    train_x = scaler.fit_transform(embeddings[train_indices]).astype(np.float32)
    y = data['y'][train_indices]
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    if min(neg, pos) == 0:
        raise ValueError('Mantis train fold lacks a class')
    model = make_model(seed, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP['learning_rate'],
                                  weight_decay=HP['weight_decay'])
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                                      torch.from_numpy(y.astype(np.float32))),
                        batch_size=HP['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    return model, loss_fn, optimizer, loader, scaler


@torch.inference_mode()
def predict(model, scaler, embeddings, indices, device):
    model.eval()
    x = scaler.transform(embeddings[indices]).astype(np.float32)
    out = []
    for start in range(0, len(indices), 256):
        batch = torch.from_numpy(x[start:start + 256]).to(device)
        out.append(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def select_epoch(data, embeddings, outer_train, fold, seed, device):
    inner_train, inner_val = cv.inner_group_split(data, outer_train, seed, fold)
    model, loss_fn, optimizer, loader, scaler = setup_training(
        data, embeddings, inner_train, seed, device)
    best_ap, best_epoch, waiting = -1.0, 0, 0
    for ep in range(1, HP['max_epochs'] + 1):
        epoch(model, loss_fn, optimizer, loader, device)
        p = predict(model, scaler, embeddings, inner_val, device)
        ap = cv.metrics(data['y'][inner_val], p)['average_precision']
        if ap > best_ap + 1e-6:
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
        parser.error('max-folds 1..5 and max-seeds 1..3 required')
    data = cv.read_source()
    folds = cv.load_folds(data)
    embeddings = load_embeddings(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Mantis E3 device:', device, '; frozen Mantis embeddings reused')
    for fold, (outer_train, outer_val) in enumerate(folds[:args.max_folds], 1):
        for seed in cv.SEEDS[:args.max_seeds]:
            if cv.load_complete('mantis', fold, seed) is not None:
                print(f'Fold {fold}, seed {seed}: already completed')
                continue
            print(f'Mantis E3 fold {fold}/5 seed {seed} – inner epoch selection', flush=True)
            start = time.perf_counter()
            epochs, inner_ap, inner_n, inner_val_n = select_epoch(
                data, embeddings, outer_train, fold, seed, device)
            print(f'Inner best epoch {epochs}; AP {inner_ap:.4f}', flush=True)
            model, loss_fn, optimizer, loader, scaler = setup_training(
                data, embeddings, outer_train, seed, device)
            for _ in range(epochs):
                epoch(model, loss_fn, optimizer, loader, device)
            p = predict(model, scaler, embeddings, outer_val, device)
            result = {'hyperparameters': HP,
                      'epoch_selected_using_inner_tic_validation': epochs,
                      'inner_best_ap': inner_ap, 'inner_train_signals': inner_n,
                      'inner_validation_signals': inner_val_n,
                      'outer_training_signals': len(outer_train),
                      'training_seconds_including_inner': round(time.perf_counter() - start, 2),
                      'metrics': cv.metrics(data['y'][outer_val], p)}
            cv.save_run('mantis', fold, seed, result, outer_val, data, p)
    print('Mantis E3 finished; original Mantis encoder and head experiments untouched.')

if __name__ == '__main__':
    main()
