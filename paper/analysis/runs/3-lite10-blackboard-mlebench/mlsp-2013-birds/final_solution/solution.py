"""MLSP 2013 birds — swarm final solution (equal-weight logit blend).

Reads frozen per-model OOF/test predictions from ./preds and writes ./submission.csv.
Run:  python solution.py
Requires: numpy, pandas, scikit-learn (only for the CV print).
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)  # sandbox root (contains data/)

# equal-weight logit ensemble (all fold-safe OOF on one canonical 5-fold split)
MODELS = [
    'lgb_bnall',             # altair_gamma:   LightGBM on hist/seg/meta + AST-PCA64 + AudioSet + BirdNET all-window stats + fold-safe neighbours
    'mil_attn2',             # pollux_beta:    attention-MIL over BirdNET window embeddings (raw+denoised) + per-window probs + BN stats + meta + neighbours
    'dn3_max',               # pollux_beta:    raw BirdNET v2.4 max prob over 45 windows (15 windows x denoise alpha 1.5/2.5/3.5), no training
    'mlp_rich_mixup',        # pollux_beta:    multi-label MLP (BCE+mixup) on 3xAST+CLAP+AudioSet+hist/mil+meta+raw-audio PCA+neighbours
]

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def main():
    oof = 1.0 / (1.0 + np.exp(-np.mean([logit(np.load(f'{HERE}/preds/oof_{n}.npy')) for n in MODELS], axis=0)))
    test = 1.0 / (1.0 + np.exp(-np.mean([logit(np.load(f'{HERE}/preds/test_{n}.npy')) for n in MODELS], axis=0)))

    lab = {}
    for line in open(f'{SB}/data/essential_data/rec_labels_test_hidden.txt'):
        line = line.strip()
        if line.startswith('rec_id'):
            continue
        p = line.split(','); rid = int(p[0])
        if p[1:] != ['?']:
            lab[rid] = tuple(int(x) for x in p[1:] if x != '')
    train_ids = sorted(lab)
    Y = np.zeros((len(train_ids), 19))
    for i, r in enumerate(train_ids):
        for s in lab[r]:
            Y[i, s] = 1
    print('final blend OOF global AUC = %.4f' % roc_auc_score(Y.ravel(), oof.ravel()))

    ss = pd.read_csv(f'{SB}/data/sample_submission.csv')
    out = pd.DataFrame({'Id': ss.Id.values, 'Probability': np.clip(test.reshape(-1), 1e-6, 1 - 1e-6)})
    out.to_csv(f'{HERE}/submission.csv', index=False)
    print('wrote', f'{HERE}/submission.csv', out.shape)

if __name__ == '__main__':
    main()
