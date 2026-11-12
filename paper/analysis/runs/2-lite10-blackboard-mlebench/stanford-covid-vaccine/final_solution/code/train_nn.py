import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

import common as C


class DS(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.Y is None:
            return torch.from_numpy(self.X[i])
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


class RNANet(nn.Module):
    def __init__(self, in_ch=14, model="gru", hidden=256, drop=0.2):
        super().__init__()
        self.model = model
        if model in ("gru", "lstm"):
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(drop),
            )
            rnn_cls = nn.GRU if model == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                128, hidden // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=drop,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 5)
            )
        else:  # dilated cnn
            blocks = []
            ch = 64
            self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU())
            for d in [1, 2, 4, 8, 16, 1, 2, 4]:
                blocks.append(ResBlock(ch, 3, d, drop))
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Conv1d(ch, 128, 3, padding=1), nn.GELU(), nn.Dropout(drop),
                nn.Conv1d(128, 5, 1),
            )

    def forward(self, x):
        # x: (B, L, C)
        x = x.permute(0, 2, 1)
        if self.model in ("gru", "lstm"):
            x = self.conv(x)
            x = x.permute(0, 2, 1)
            x, _ = self.rnn(x)
            return self.head(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.permute(0, 2, 1)


class ResBlock(nn.Module):
    def __init__(self, ch, k, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.b1 = nn.BatchNorm1d(ch)
        self.b2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.drop(h)
        h = self.b2(self.c2(h))
        return self.act(x + h)


def augment_reverse(X, Y, p=0.5):
    if np.random.rand() < p:
        X = X[:, ::-1, :].copy()
        Y = Y[:, ::-1, :].copy()
    return X, Y


def train_one(args, Xtr, Ytr, Xva, Yva, Xte, seed, va_sn=None, tr_w=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNANet(model=args.model, hidden=args.hidden, drop=args.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_ep = args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=n_ep, steps_per_epoch=max(1, len(Xtr) // args.bs)
    )
    crit = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss(beta=0.5)
    tr_ds = DS(Xtr, Ytr)
    if tr_w is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(tr_w, dtype=torch.double), num_samples=len(tr_w), replacement=True)
        dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, drop_last=True, num_workers=0)
    else:
        dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
    va_x = torch.from_numpy(Xva).to(dev)
    te_x = torch.from_numpy(Xte).to(dev)
    L = Ytr.shape[1]
    best = 1e9
    best_state = None
    best_ep = -1
    patience = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    for ep in range(n_ep):
        model.train()
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            if args.aug:
                xb, yb = augment_reverse(xb, yb)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                p = model(xb)
                if args.scored_only:
                    idx3 = [0, 1, 3]
                    loss = crit(p[:, :L, idx3], yb[:, :, idx3])
                else:
                    loss = crit(p[:, :L, :], yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            pv = model(va_x)[:, :L, :].float().cpu().numpy()
        if args.select == "sn" and va_sn is not None and va_sn.sum() > 0:
            m, _ = C.mcrmse(Yva[va_sn], pv[va_sn])
        else:
            m, _ = C.mcrmse(Yva, pv)
        if m < best - 1e-5:
            best = m
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev == "cuda")):
        pv = model(va_x)[:, :L, :].float().cpu().numpy()
        pt = model(te_x).float().cpu().numpy()
    return pv, pt, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--select", default="sn", choices=["sn","all"])
    ap.add_argument("--scored-only", action="store_true")
    ap.add_argument("--train-sn1", action="store_true")
    ap.add_argument("--sn-weight", action="store_true")
    ap.add_argument("--sn-cap", type=float, default=100.0)
    ap.add_argument("--sn-power", type=float, default=1.0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    tr = C.load_train()
    te = C.load_test()
    X = C.make_arrays(tr)
    Y = C.make_targets(tr)
    Xte = C.make_arrays(te)
    sn = np.array([r["SN_filter"] for r in tr], dtype=bool)
    print("train", X.shape, "test", Xte.shape, "SN1", sn.sum(), flush=True)

    folds = C.kfold_indices(len(tr), args.folds, 42)
    oof = np.zeros_like(Y)
    te_preds = []
    t0 = time.time()
    fold_scores = []
    for k, (tidx, vidx) in enumerate(folds):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        if args.train_sn1:
            tidx = tidx[sn[tidx]]
        tr_w = None
        if args.sn_weight:
            _w = np.array([max(0.05, r["signal_to_noise"]) for r in tr], dtype=np.float32)
            _w = np.minimum(_w, args.sn_cap) ** args.sn_power
            tr_w = _w[tidx]
        pv, pt, bv, bep = train_one(args, X[tidx], Y[tidx], X[vidx], Y[vidx], Xte, args.seed + k, va_sn=sn[vidx], tr_w=tr_w)
        oof[vidx] = pv
        te_preds.append(pt)
        m_all, _ = C.mcrmse(Y[vidx], pv)
        m_sn, _ = C.mcrmse(Y[vidx][sn[vidx]], pv[sn[vidx]])
        fold_scores.append((m_all, m_sn))
        print(f"fold {k}: all={m_all:.4f} sn1={m_sn:.4f} best_ep={bep} best_val={bv:.4f} t={time.time()-t0:.0f}s", flush=True)

    m_all, rms = C.mcrmse(Y, oof)
    m_sn, rms_sn = C.mcrmse(Y[sn], oof[sn])
    print(f"OOF all={m_all:.5f} {rms}", flush=True)
    print(f"OOF sn1={m_sn:.5f} {rms_sn}", flush=True)
    te_mean = np.mean(te_preds, axis=0)
    out = f"pred_{args.tag}"
    np.save(out + "_oof.npy", oof.astype(np.float32))
    np.save(out + "_test.npy", te_mean.astype(np.float32))
    with open(out + "_meta.json", "w") as f:
        json.dump({"args": vars(args), "oof_all": m_all, "oof_sn1": m_sn,
                   "rms_all": rms, "rms_sn": rms_sn, "folds": fold_scores}, f, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
