import os, sys, time, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

CACHE = 'workspaces/sirius_tau/cache'
SEQ = 80


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='lstm', choices=['lstm', 'transformer', 'gru'])
    p.add_argument('--hidden', type=int, default=384)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--drop', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp_w', type=float, default=0.1, help='loss weight for expiratory rows')
    p.add_argument('--tag', default='exp')
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--nf_dim', type=int, default=256)
    p.add_argument('--cache', default=CACHE)
    p.add_argument('--sched', default='plateau', choices=['plateau', 'cosine'])
    p.add_argument('--init', default='')
    return p.parse_args()


class LSTMNet(nn.Module):
    def __init__(self, nf, hidden, layers, drop, rnn='lstm'):
        super().__init__()
        cls = nn.LSTM if rnn == 'lstm' else nn.GRU
        self.rnn = cls(nf, hidden, layers, batch_first=True, bidirectional=True,
                       dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))
        self.skip = nn.Linear(nf, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1) + self.skip(x).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, ch, k, dil, drop):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, k, padding='same', dilation=dil)
        self.conv2 = nn.Conv1d(ch, ch, k, padding='same', dilation=dil)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(torch.nn.functional.gelu(self.n1(self.conv1(x))))
        h = self.drop(torch.nn.functional.gelu(self.n2(self.conv2(h))))
        return x + h


class TCNNet(nn.Module):
    def __init__(self, nf, hidden, layers, drop, **kw):
        super().__init__()
        self.stem = nn.Conv1d(nf, hidden, 1)
        dils = [2 ** i for i in range(layers)]
        self.blocks = nn.Sequential(*[TCNBlock(hidden, 3, d, drop) for d in dils])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
        self.skip = nn.Linear(nf, 1)

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        h = self.blocks(h)
        h = h.transpose(1, 2)
        return self.head(h).squeeze(-1) + self.skip(x).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, nf, hidden, layers, drop, heads, nf_dim):
        super().__init__()
        self.proj = nn.Linear(nf, hidden)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=nf_dim,
                                           dropout=drop, batch_first=True, activation='gelu',
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
        self.skip = nn.Linear(nf, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.enc(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1) + self.skip(x).squeeze(-1)


def predict(model, X, bs=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            outs.append(model(X[i:i + bs]).float().cpu())
    return torch.cat(outs, 0).numpy()


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = 'cuda'
    outdir = f'workspaces/sirius_tau/runs/{a.tag}'
    os.makedirs(outdir, exist_ok=True)

    X = np.load(f'{a.cache}/X_train.npy', mmap_mode='r')
    y = np.load(f'{a.cache}/y_train.npy')
    uout_idx = open(f'{a.cache}/feature_names.txt').read().split('\n').index('u_out')
    trn_idx = np.load(f'{a.cache}/trn_idx.npy')
    val_idx = np.load(f'{a.cache}/val_idx.npy')
    Xte = np.load(f'{a.cache}/X_test.npy', mmap_mode='r')

    Xtr = np.asarray(X[trn_idx], dtype=np.float32)
    ytr = np.asarray(y[trn_idx], dtype=np.float32)
    Xva = np.asarray(X[val_idx], dtype=np.float32)
    yva = np.asarray(y[val_idx], dtype=np.float32)

    uotr_raw = Xtr[:, :, uout_idx].copy()
    uova_raw = Xva[:, :, uout_idx].copy()
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    Xte_n = ((np.asarray(Xte, dtype=np.float32) - mean) / std).astype(np.float32)

    Xtr_g = torch.from_numpy(Xtr).to(dev)
    ytr_g = torch.from_numpy(ytr).to(dev)
    uotr_g = torch.from_numpy(uotr_raw).to(dev)
    Xva_g = torch.from_numpy(Xva).to(dev)
    yva_g = torch.from_numpy(yva).to(dev)
    uova_g = torch.from_numpy(uova_raw).to(dev)
    nf = Xtr.shape[-1]

    if a.model in ('lstm', 'gru'):
        model = LSTMNet(nf, a.hidden, a.layers, a.drop, a.model).to(dev)
    elif a.model == 'tcn':
        model = TCNNet(nf, a.hidden, a.layers, a.drop).to(dev)
    else:
        model = TransformerNet(nf, a.hidden, a.layers, a.drop, a.heads, a.nf_dim).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        print(f'[{a.tag}] initialized from {a.init}', flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[{a.tag}] model={a.model} params={nparam/1e6:.2f}M nf={nf} seed={a.seed}', flush=True)

    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.sched == 'cosine':
        sched = CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    else:
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=6, min_lr=1e-6)
    ntr = len(trn_idx)
    best = 1e9
    best_state = None
    wait = 0
    log = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(ntr, device=dev)
        tot = 0.0
        nb = 0
        for i in range(0, ntr, a.bs):
            idx = perm[i:i + a.bs]
            xb = Xtr_g[idx]; yb = ytr_g[idx]; wb = uotr_g[idx]
            pred = model(xb)
            w = torch.where(wb == 0, 1.0, a.exp_w)
            loss = (w * (pred - yb).abs()).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        # validation
        pv = predict(model, Xva_g)
        m_insp = float(np.abs(pv - yva)[uova_g.cpu().numpy() == 0].mean())
        m_all = float(np.abs(pv - yva).mean())
        if a.sched == 'cosine':
            sched.step()
        else:
            sched.step(m_insp)
        lr = opt.param_groups[0]['lr']
        log.append((ep, tot / nb, m_insp, m_all, lr))
        if ep % 5 == 0 or ep == 1:
            print(f'[{a.tag}] ep{ep:3d} loss={tot/nb:.4f} val_insp={m_insp:.5f} val_all={m_all:.5f} lr={lr:.2e} t={time.time()-t0:.0f}s', flush=True)
        if m_insp < best - 1e-6:
            best = m_insp; wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                print(f'[{a.tag}] early stop ep{ep}', flush=True)
                break

    model.load_state_dict(best_state)
    pv = predict(model, Xva_g)
    best_insp = float(np.abs(pv - yva)[uova_g.cpu().numpy() == 0].mean())
    best_all = float(np.abs(pv - yva).mean())
    print(f'[{a.tag}] BEST val_insp={best_insp:.5f} val_all={best_all:.5f} elapsed={time.time()-t0:.0f}s', flush=True)

    # save val preds (breath x seq)
    np.save(f'{outdir}/val_pred.npy', pv.astype(np.float32))
    # test predictions
    Xte_g = torch.from_numpy(Xte_n).to(dev)
    pt = predict(model, Xte_g)
    torch.save(best_state, f'{outdir}/model.pt')
    # write per-row submission aligned to test.csv order via id
    te = pd.read_csv('data/test.csv', usecols=['id', 'breath_id', 'time_step'])
    # order of Xte breaths corresponds to sorted(breath_id,time_step) global; we saved test breath ids in that order
    tbid = np.load(f'{a.cache}/test_breath_ids.npy')
    flat_pred = pt.reshape(-1)
    key = pd.DataFrame({'breath_id': np.repeat(tbid, SEQ),
                        'time_step': np.tile(np.arange(SEQ), len(tbid)),
                        'pressure': flat_pred})
    # map using breath_id + rank of time_step within breath (time_step values are unique within breath, sorted)
    te_sorted = te.sort_values(['breath_id', 'time_step'], kind='mergesort').reset_index(drop=True)
    assert len(te_sorted) == len(key)
    assert (te_sorted['breath_id'].values == key['breath_id'].values).all()
    sub = pd.DataFrame({'id': te_sorted['id'].values, 'pressure': key['pressure'].values})
    sample = pd.read_csv('data/sample_submission.csv')
    sub = sample[['id']].merge(sub, on='id', how='left')
    assert sub['pressure'].notna().all()
    sub.to_csv(f'{outdir}/test_pred.csv', index=False)
    np.save(f'{outdir}/val_idx.npy', val_idx)
    json.dump({'tag': a.tag, 'model': a.model, 'val_insp': best_insp, 'val_all': best_all,
               'args': vars(a), 'params': nparam, 'elapsed': time.time() - t0,
               'log': log}, open(f'{outdir}/result.json', 'w'), indent=2)
    print('saved', outdir, flush=True)


if __name__ == '__main__':
    main()
