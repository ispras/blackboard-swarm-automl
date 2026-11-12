import os, sys, argparse, json, time
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LSTMNet, predict, SEQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--exp_w', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'; os.makedirs(a.out, exist_ok=True)

    names = open(f'{a.cache}/feature_names.txt').read().split('\n')
    uoi = names.index('u_out')
    X = np.load(f'{a.cache}/X_train.npy', mmap_mode='r')
    y = np.load(f'{a.cache}/y_train.npy')
    trn_idx = np.load(f'{a.cache}/trn_idx.npy')
    nf = len(names)
    # keep SAME normalization as base training (stats over the original 90% split)
    Xstat = np.asarray(X[trn_idx], dtype=np.float32)
    mean = Xstat.reshape(-1, nf).mean(0); std = Xstat.reshape(-1, nf).std(0) + 1e-6
    del Xstat
    Xall = ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)
    uo = np.asarray(X[:, :, uoi]).copy()
    del X

    Xg = torch.from_numpy(Xall).to(dev)
    yg = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
    uog = torch.from_numpy(uo).to(dev)
    n = Xg.shape[0]

    model = LSTMNet(nf, a.hidden, a.layers, a.drop, 'lstm').to(dev)
    model.load_state_dict(torch.load(a.init, map_location=dev))
    print(f'loaded init, training full data n={n}', flush=True)
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0; nb = 0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            xb = Xg[idx]; yb = yg[idx]; wb = uog[idx]
            pred = model(xb)
            w = torch.where(wb == 0, 1.0, a.exp_w)
            loss = (w * (pred - yb).abs()).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f'full ep{ep} loss={tot/nb:.4f} lr={opt.param_groups[0]["lr"]:.2e} t={time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{a.out}/model.pt')

    Xte = ((np.asarray(np.load(f'{a.cache}/X_test.npy', mmap_mode='r'), dtype=np.float32) - mean) / std).astype(np.float32)
    pt = predict(model, torch.from_numpy(Xte).to(dev))
    tbid = np.load(f'{a.cache}/test_breath_ids.npy')
    te = pd.read_csv('data/test.csv', usecols=['id', 'breath_id', 'time_step'])
    te_sorted = te.sort_values(['breath_id', 'time_step'], kind='mergesort').reset_index(drop=True)
    assert (te_sorted['breath_id'].values == np.repeat(tbid, SEQ)).all()
    sub = pd.DataFrame({'id': te_sorted['id'].values, 'pressure': pt.reshape(-1)})
    sample = pd.read_csv('data/sample_submission.csv')
    sub = sample[['id']].merge(sub, on='id', how='left')
    assert sub['pressure'].notna().all()
    sub.to_csv(f'{a.out}/test_pred.csv', index=False)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
