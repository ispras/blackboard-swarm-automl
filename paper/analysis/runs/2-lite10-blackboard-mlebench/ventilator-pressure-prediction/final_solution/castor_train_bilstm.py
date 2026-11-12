import argparse, json, os, time, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CACHE = "/download_job_code/payload/sandbox/workspaces/castor_eta/cache"
OUT = "/download_job_code/payload/sandbox/workspaces/castor_eta/runs"
os.makedirs(OUT, exist_ok=True)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arch", default="cnn_gru", choices=["cnn_gru", "bilstm", "gru"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--conv", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_w", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--folds", default="1,2,3,4")
    p.add_argument("--use_all", action="store_true", help="train on all folds, no early stop")
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--snap", type=int, default=1, help="average weights of last N epochs")
    p.add_argument("--cache", default=CACHE)
    return p.parse_args()


class CNNGRU(nn.Module):
    def __init__(self, F, hidden, layers, dropout, conv):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(F, conv, 3, padding=1), nn.GELU(),
            nn.Conv1d(conv, conv, 3, padding=1), nn.GELU(),
        )
        self.rnn = nn.GRU(conv, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h).squeeze(-1)


class RNN(nn.Module):
    def __init__(self, F, hidden, layers, dropout, kind="lstm"):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(F, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)


def build_model(a, F):
    if a.arch == "cnn_gru":
        return CNNGRU(F, a.hidden, a.layers, a.dropout, a.conv)
    if a.arch == "bilstm":
        return RNN(F, a.hidden, a.layers, a.dropout, "lstm")
    return RNN(F, a.hidden, a.layers, a.dropout, "gru")


def run_epoch(model, dl, opt, sched, crit, device, train=True):
    model.train(train)
    tot, n, se, si, ni, ntot = 0.0, 0, 0.0, 0.0, 0.0, 0.0
    for xb, yb, mb in dl:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = crit(pred, yb, mb)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        tot += loss.item() * len(xb); n += len(xb)
        with torch.no_grad():
            d = (pred - yb).abs()
            se += (d * mb).sum().item(); si += mb.sum().item()
    return tot / n, se / max(si, 1)


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"
    X = np.load(f"{a.cache}/X_train.npy"); y = np.load(f"{a.cache}/y_train.npy")
    m = np.load(f"{a.cache}/m_train.npy"); fold = np.load(f"{a.cache}/fold_shared.npy")
    Xte = np.load(f"{a.cache}/X_test.npy")
    F = X.shape[-1]
    if a.use_all:
        tri = np.arange(len(X)); vai = np.where(fold == 0)[0]
    else:
        train_folds = [int(v) for v in a.folds.split(",")]
        tri = np.where(np.isin(fold, train_folds))[0]; vai = np.where(fold == 0)[0]
    print(f"train {len(tri)} val {len(vai)} F {F} arch {a.arch}", flush=True)
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X[tri]), torch.from_numpy(y[tri]),
                                     torch.from_numpy(m[tri])), batch_size=a.batch,
                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X[vai]), torch.from_numpy(y[vai]),
                                     torch.from_numpy(m[vai])), batch_size=1024,
                       shuffle=False, num_workers=4, pin_memory=True)
    model = build_model(a, F).to(device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)

    def crit(pred, yb, mb):
        w = a.exp_w + (1.0 - a.exp_w) * mb
        return ((pred - yb).abs() * w).sum() / w.sum()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(dl_tr) * a.epochs; warm = len(dl_tr) * a.warm
    def lrf(s):
        c = 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        return a.min_lr / a.lr + (1 - a.min_lr / a.lr) * c * min(1.0, (s + 1) / warm)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrf)

    best = 1e9; best_ep = -1; bad = 0; hist = []
    snaps = []
    ckpt = f"{OUT}/{a.name}.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        ltr, atr = run_epoch(model, dl_tr, opt, sched, crit, device, True)
        lva, ava = run_epoch(model, dl_va, None, None, crit, device, False)
        hist.append({"ep": ep, "train_loss": ltr, "train_insp": atr, "val_loss": lva, "val_insp": ava})
        marker = ""
        if a.use_all:
            best = ava; best_ep = ep
            torch.save({"sd": model.state_dict(), "args": vars(a), "F": F, "best": best}, ckpt)
            marker = " (last)"
        elif ava < best - 1e-5:
            best = ava; best_ep = ep; bad = 0
            torch.save({"sd": model.state_dict(), "args": vars(a), "F": F, "best": best}, ckpt)
            marker = " *"
        else:
            bad += 1
        if ep % 1 == 0 or marker:
            print(f"ep{ep:3d} trL {ltr:.4f} trI {atr:.4f} | vaL {lva:.4f} vaI {ava:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e}{marker} [{time.time()-t0:.0f}s]", flush=True)
        if a.snap > 1:
            snaps.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
            snaps = snaps[-a.snap:]
        if (not a.use_all) and bad >= a.patience:
            print("early stop", flush=True); break

    if a.snap > 1 and snaps:
        avg = {k: torch.stack([sd[k].float() for sd in snaps]).mean(0) for k in snaps[0]}
        print(f"snapshot-avg last {len(snaps)} epochs", flush=True)
        model.load_state_dict(avg)
        torch.save({"sd": avg, "args": vars(a), "F": F, "best": best}, ckpt)
    else:
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["sd"])
    model.eval()

    def predict(A, bs=2048):
        dl = DataLoader(TensorDataset(torch.from_numpy(A)), batch_size=bs, shuffle=False,
                        num_workers=4, pin_memory=True)
        out = []
        with torch.no_grad():
            for (xb,) in dl:
                out.append(model(xb.to(device)).float().cpu().numpy())
        return np.concatenate(out)

    pv = predict(X[vai]); pt = predict(Xte)
    d = np.abs(pv - y[vai])
    val_insp = float((d * m[vai]).sum() / m[vai].sum())
    val_all = float(d.mean())
    print(f"BEST ep {best_ep} val_insp_MAE {val_insp:.5f} val_all_MAE {val_all:.5f}", flush=True)
    np.save(f"{OUT}/{a.name}_val.npy", pv)
    np.save(f"{OUT}/{a.name}_val_idx.npy", vai)
    np.save(f"{OUT}/{a.name}_test.npy", pt)
    json.dump({"args": vars(a), "best_ep": best_ep, "val_insp_mae": val_insp,
               "val_all_mae": val_all, "history": hist},
              open(f"{OUT}/{a.name}_meta.json", "w"), indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
