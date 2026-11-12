import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import common as C


PAIRVAL = {"AU": 1, "UA": 1, "GC": 2, "CG": 2, "GU": 3, "UG": 3}


def contact_map(struct, seq, L=107):
    Cm = np.zeros((L, L, 3), dtype=np.float32)
    stack = []
    for i, ch in enumerate(struct[:L]):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            Cm[i, j, 0] = Cm[j, i, 0] = 1.0
            pv = PAIRVAL.get(seq[i] + seq[j], 0) / 3.0
            Cm[i, j, 1] = Cm[j, i, 1] = pv
            d = abs(i - j) / L
            Cm[i, j, 2] = Cm[j, i, 2] = d
    return Cm


LOOPS = list("SMI BHEX".replace(" ", ""))
BASE = {b: i for i, b in enumerate("ACGU")}


def partner_idx(struct, L=107):
    p = np.full(L, -1, dtype=np.int64)
    stack = []
    for i, ch in enumerate(struct[:L]):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            p[i] = j
            p[j] = i
    return p


def make_struct_arrays(records, L=107):
    base = C.make_arrays(records)
    X = np.zeros((len(records), L, 14 + 12), dtype=np.float32)
    Cm = np.zeros((len(records), L, L, 3), dtype=np.float32)
    for i, r in enumerate(records):
        Cm[i] = contact_map(r["structure"], r["sequence"], L)
        p = partner_idx(r["structure"], L)
        ex = np.zeros((L, 12), dtype=np.float32)
        seq, lp = r["sequence"], r["predicted_loop_type"]
        for k in range(L):
            j = p[k]
            if j >= 0:
                ex[k, 0] = 1.0
                ex[k, 1 + BASE.get(seq[j], 0)] = 1.0
                g = LOOPS.index(lp[j]) if j < len(lp) and lp[j] in LOOPS else 0
                ex[k, 5 + g] = 1.0
        X[i, :, :14] = base[i]
        X[i, :, 14:] = ex
    return X, Cm


class DS(Dataset):
    def __init__(self, X, Cm, Y=None):
        self.X, self.Cm, self.Y = X, Cm, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        c = torch.from_numpy(self.Cm[i])
        if self.Y is None:
            return x, c
        return x, c, torch.from_numpy(self.Y[i])


class Res2d(nn.Module):
    def __init__(self, ch, drop):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(self.act(self.b1(self.c1(x))))
        h = self.b2(self.c2(h))
        return self.act(x + h)


class StructNet(nn.Module):
    def __init__(self, in_ch=14, hidden=512, drop=0.2, ch2=32):
        super().__init__()
        ch = 64
        self.convs = nn.ModuleList([nn.Conv1d(in_ch, ch, k, padding=k // 2) for k in (3, 5, 9)])
        self.bn1 = nn.BatchNorm1d(ch * 3)
        self.act = nn.GELU()
        self.drop0 = nn.Dropout(drop)
        self.s2 = nn.Sequential(
            nn.Conv2d(3, ch2, 3, padding=1), nn.BatchNorm2d(ch2), nn.GELU(),
            Res2d(ch2, drop), Res2d(ch2, drop),
        )
        self.p2 = nn.Sequential(nn.Conv1d(ch2, 48, 3, padding=1), nn.BatchNorm1d(48), nn.GELU())
        self.rnn = nn.GRU(ch * 3 + 48, hidden // 2, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 5))

    def forward(self, x, cm):
        x = x.permute(0, 2, 1)  # B,C,L
        hs = [self.act(c(x)) for c in self.convs]
        h1 = self.drop0(self.act(self.bn1(torch.cat(hs, 1))))  # B,192,L
        h2 = self.s2(cm.permute(0, 3, 1, 2))  # B,ch2,L,L
        h2 = h2.sum(-1) / (h2.shape[-1] ** 0.5)  # B,ch2,L  (partner-context pooled)
        h2 = self.p2(h2)
        h = torch.cat([h1, h2], 1).permute(0, 2, 1)  # B,L,240
        h, _ = self.rnn(h)
        return self.head(h)


def train_one(args, Xtr, Ctr, Ytr, Xva, Cva, Yva, Xte, Cte, seed, va_sn=None, tr_w=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = StructNet(in_ch=args.in_ch, hidden=args.hidden, drop=args.drop, ch2=args.ch2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_ep = args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=n_ep,
                                                steps_per_epoch=max(1, len(Xtr) // args.bs))
    crit = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss(beta=0.5)
    ds = DS(Xtr, Ctr, Ytr)
    if tr_w is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(tr_w, dtype=torch.double),
                                                         num_samples=len(tr_w), replacement=True)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, drop_last=True, num_workers=0)
    else:
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
    va_x, va_c = torch.from_numpy(Xva).to(dev), torch.from_numpy(Cva).to(dev)
    te_x, te_c = torch.from_numpy(Xte).to(dev), torch.from_numpy(Cte).to(dev)
    L = Ytr.shape[1]
    best, best_state, best_ep, patience = 1e9, None, -1, 0
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    for ep in range(n_ep):
        model.train()
        for xb, cb, yb in dl:
            xb, cb, yb = xb.to(dev), cb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                p = model(xb, cb)
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
            pv = model(va_x, va_c)[:, :L, :].float().cpu().numpy()
        m = C.mcrmse(Yva[va_sn], pv[va_sn])[0] if (va_sn is not None and va_sn.sum() > 0) else C.mcrmse(Yva, pv)[0]
        if m < best - 1e-5:
            best, best_ep, patience = m, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev == "cuda")):
        pv = model(va_x, va_c)[:, :L, :].float().cpu().numpy()
        pt = model(te_x, te_c).float().cpu().numpy()
    return pv, pt, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--in-ch", type=int, default=26)
    ap.add_argument("--ch2", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--scored-only", action="store_true")
    ap.add_argument("--sn-weight", action="store_true")
    ap.add_argument("--tag", default="struct")
    args = ap.parse_args()

    tr = C.load_train()
    te = C.load_test()
    X, Cm = make_struct_arrays(tr)
    Xte, Cm_te = make_struct_arrays(te)
    Y = C.make_targets(tr)
    sn = np.array([r["SN_filter"] for r in tr], dtype=bool)
    print("train", X.shape, Cm.shape, "test", Xte.shape, "SN1", sn.sum(), flush=True)

    folds = C.kfold_indices(len(tr), 5, 42)
    oof = np.zeros_like(Y)
    te_preds = []
    t0 = time.time()
    for k, (tidx, vidx) in enumerate(folds):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        tr_w = None
        if args.sn_weight:
            _w = np.minimum(np.array([max(0.05, r["signal_to_noise"]) for r in tr], dtype=np.float32), 100.0)
            tr_w = _w[tidx]
        pv, pt, bv, bep = train_one(args, X[tidx], Cm[tidx], Y[tidx], X[vidx], Cm[vidx], Y[vidx],
                                    Xte, Cm_te, args.seed + k, va_sn=sn[vidx], tr_w=tr_w)
        oof[vidx] = pv
        te_preds.append(pt)
        m_all = C.mcrmse(Y[vidx], pv)[0]
        m_sn = C.mcrmse(Y[vidx][sn[vidx]], pv[sn[vidx]])[0]
        print(f"fold {k}: all={m_all:.4f} sn1={m_sn:.4f} best_ep={bep} best_val={bv:.4f} t={time.time()-t0:.0f}s", flush=True)

    m_sn, rms_sn = C.mcrmse(Y[sn], oof[sn])
    print(f"OOF sn1={m_sn:.5f} {rms_sn}", flush=True)
    te_mean = np.mean(te_preds, axis=0)
    out = f"pred_{args.tag}"
    np.save(out + "_oof.npy", oof.astype(np.float32))
    np.save(out + "_test.npy", te_mean.astype(np.float32))
    with open(out + "_meta.json", "w") as f:
        json.dump({"args": vars(args), "oof_sn1": m_sn, "rms_sn": rms_sn}, f, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
