import argparse, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = "/download_job_code/payload/sandbox"
SH = f"{ROOT}/workspaces/shared"
OUT = f"{ROOT}/workspaces/canopus_iota"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Net(nn.Module):
    def __init__(self, nf, hidden=256, layers=3, dropout=0.1, head=128, arch="lstm"):
        super().__init__()
        RNN = nn.LSTM if arch == "lstm" else nn.GRU
        self.lstm = RNN(nf, hidden, num_layers=layers, batch_first=True,
                        bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, head), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def load_data(feat_path=None, norm_path=None):
    d = np.load(feat_path or f"{SH}/ventilator_v1.npz", allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    y = d["y_train"].astype(np.float32)
    uout = d["u_out_train"].astype(np.float32)
    bid_tr = d["breath_id_train"]
    Xte = d["X_test"].astype(np.float32)
    bid_te = d["breath_id_test"]
    if norm_path:
        nm = np.load(norm_path)
        mu, sd = nm["mean"].astype(np.float32), nm["std"].astype(np.float32)
    else:
        flat = X.reshape(-1, X.shape[2])
        mu = flat.mean(0)
        sd = flat.std(0) + 1e-6
    X = (X - mu) / sd
    Xte = (Xte - mu) / sd
    folds = pd.read_csv(f"{SH}/folds.csv")
    fmap = dict(zip(folds.breath_id, folds.fold))
    fold_of = np.array([fmap[b] for b in bid_tr], dtype=np.int16)
    return X, y, uout, bid_tr, Xte, bid_te, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head", type=int, default=128)
    ap.add_argument("--arch", default="lstm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--exp_w", type=float, default=0.3)
    ap.add_argument("--tag", default="lstm")
    ap.add_argument("--feat", default="")
    ap.add_argument("--norm", default="")
    ap.add_argument("--snap_last", type=int, default=5)
    ap.add_argument("--save_test", type=int, default=1)
    ap.add_argument("--alldata", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    t0 = time.time()

    X, y, uout, bid_tr, Xte, bid_te, fold_of = load_data(args.feat or None, args.norm or None)
    insp = (uout < 0.5)
    w = np.where(insp, 1.0, args.exp_w).astype(np.float32)

    dev = "cuda"
    Xg = torch.from_numpy(X).to(dev)
    yg = torch.from_numpy(y).to(dev)
    wg = torch.from_numpy(w).to(dev)
    Xteg = torch.from_numpy(Xte).to(dev)
    del X, Xte

    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_test = []
    fold_stats = {}
    for f in [int(x) for x in args.folds.split(",")]:
        if args.alldata:
            tr_idx = np.arange(len(fold_of))
            va_idx = np.where(fold_of == f)[0]
        else:
            tr_idx = np.where(fold_of != f)[0]
            va_idx = np.where(fold_of == f)[0]
        trt = torch.from_numpy(tr_idx).to(dev)
        vat = torch.from_numpy(va_idx).to(dev)
        net = Net(Xg.shape[2], args.hidden, args.layers, args.dropout, args.head, args.arch).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=2e-5)
        scaler = torch.cuda.amp.GradScaler()
        n = len(tr_idx)
        snap_v, snap_t = [], []
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for i in range(0, n, args.bs):
                b = trt[perm[i:i + args.bs]]
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    p = net(Xg[b])
                    loss = (torch.abs(p - yg[b]) * wg[b]).sum() / wg[b].sum()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += float(loss) * len(b)
            sched.step()
            do_snap = ep >= args.epochs - args.snap_last
            if do_snap or ep % 10 == 0 or ep == args.epochs - 1:
                net.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    pv = net(Xg[vat]).float().cpu().numpy()
                if do_snap:
                    snap_v.append(pv)
                    if args.save_test:
                        tp = []
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            for i in range(0, len(Xteg), 2048):
                                tp.append(net(Xteg[i:i + 2048]).float().cpu().numpy())
                        snap_t.append(np.concatenate(tp, 0))
                if ep % 10 == 0 or ep == args.epochs - 1:
                    yv = y[va_idx]
                    mi = float(np.abs(pv - yv)[insp[va_idx]].mean())
                    ma = float(np.abs(pv - yv).mean())
                    print(f"f{f} ep{ep} loss {tot/n:.5f} val_insp {mi:.5f} val_all {ma:.5f} t{time.time()-t0:.0f}s", flush=True)
        pv = np.mean(snap_v, 0)
        yv = y[va_idx]
        m_all = float(np.abs(pv - yv).mean())
        m_ins = float(np.abs(pv - yv)[insp[va_idx]].mean())
        m_exp = float(np.abs(pv - yv)[~insp[va_idx]].mean())
        if not args.alldata:
            oof[va_idx] = pv
        if args.save_test:
            fold_test.append(np.mean(snap_t, 0))
        fold_stats[f] = (m_ins, m_all, m_exp)
        print(f"FOLD {f} snap MAE_insp {m_ins:.5f} MAE_all {m_all:.5f} MAE_exp {m_exp:.5f}", flush=True)
        # incremental checkpoint after every fold (crash safety)
        if not args.alldata:
            np.save(f"{OUT}/oof_{args.tag}.npy", oof)
        if args.save_test and fold_test:
            np.save(f"{OUT}/test_pred_{args.tag}.npy", np.mean(fold_test, 0))
        json.dump({"args": vars(args), "fold_stats": {str(k): v for k, v in fold_stats.items()},
                   "partial": True}, open(f"{OUT}/args_{args.tag}.json", "w"), indent=1)

    tag = args.tag
    if not args.alldata:
        np.save(f"{OUT}/oof_{tag}.npy", oof)
    if args.save_test and fold_test:
        np.save(f"{OUT}/test_pred_{tag}.npy", np.mean(fold_test, 0))
    ins_mae = None if args.alldata else (float(np.abs(oof - y)[insp].mean()) if not np.isnan(oof).all() else None)
    print("RUN", tag, "OOF_insp", None if ins_mae is None else round(ins_mae, 5), "folds", fold_stats, flush=True)
    json.dump({"args": vars(args), "fold_stats": {str(k): v for k, v in fold_stats.items()},
               "oof_insp": ins_mae, "secs": time.time() - t0}, open(f"{OUT}/args_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
