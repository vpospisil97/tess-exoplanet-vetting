"""E3: five-fold TIC-safe CV for existing logistic-regression baseline.
Run after prepare_e3_folds.py; no held-out-test fitting or scoring.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_e3_folds as cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-folds', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5:
        parser.error('--max-folds must be 1..5')
    data = cv.read_source()
    folds = cv.load_folds(data)
    features = np.concatenate((data['local'].reshape(len(data['y']), -1),
                               data['global'].reshape(len(data['y']), -1)), axis=1)
    if features.shape != (1918, 604):
        raise RuntimeError(f'Unexpected baseline feature shape: {features.shape}')
    for fold, (tr, va) in enumerate(folds[:args.max_folds], 1):
        if cv.load_complete('baseline', fold, None) is not None:
            print(f'Fold {fold}: already completed')
            continue
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=0.1, class_weight='balanced',
                                              solver='liblinear', max_iter=4000,
                                              random_state=42)),
        ])
        model.fit(features[tr], data['y'][tr])
        p = model.predict_proba(features[va])[:, 1]
        result = {'hyperparameters': {'C': 0.1, 'class_weight': 'balanced',
                                      'solver': 'liblinear', 'max_iter': 4000},
                  'metrics': cv.metrics(data['y'][va], p)}
        cv.save_run('baseline', fold, None, result, va, data, p)
    print('Baseline E3 complete for requested folds; original artifacts unchanged.')

if __name__ == '__main__':
    main()
