import os, sys, time, argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "workspaces/alkaid_psi"
S = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


class PairDS(Dataset):
    def __init__(self, split, idx, train):
        self.split = split
        self.idx = np.asarray(idx, np.int64)
        self.train = train
        self.arr = np.load(f"{OUT}/{split}_crops.npy", mmap_mode="r")
        self.geom = np.load(f"{OUT}/geom_{split}.npy", mmap_mode="r")
        p = pd.read_parquet(f"{OUT}/{split}_cand.parquet")
        self.y = p.contact.values.astype(np.float32) if "contact" in p else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = int(self.idx[i])
        x = np.asarray(self.arr[r])  # (2,S,S,3) uint8
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                f = np.float32(np.random.uniform(0.85, 1.15))
                b = np.float32(np.random.uniform(-15, 15))
                x = np.clip(x.astype(np.float32) * f + b, 0, 255).astype(np.uint8)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))  # (2,3,S,S)
        g = np.asarray(self.geom[r]).copy()
        y = self.y[r] if self.y is not None else 0.0
        return torch.from_numpy(x), torch.from_numpy(g), torch.tensor(y)


class Net(nn.Module):
    def __init__(self, backbone, gn, nviews=2):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0, in_chans=3)
        nf = self.bb.num_features
        self.nf = nf
        self.nviews = nviews
        self.head = nn.Sequential(
            nn.Linear(nviews * nf + gn, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, x, g):
        B, V = x.shape[0], x.shape[1]
        f = self.bb(x.reshape(B * V, *x.shape[2:])).reshape(B, V * self.nf)
        return self.head(torch.cat([f, g], 1)).squeeze(1)


def normalize(x):
    x = x.float() / 255.0
    m = torch.tensor(MEAN / 255.0, device=x.device).view(1, 1, 3, 1, 1)
    s = torch.tensor(STD / 255.0, device=x.device).view(1, 1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ps = []
    for x, g, y in dl:
        x = normalize(x.to(device, non_blocking=True))
        g = g.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            z = model(x, g)
        ps.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(ps)


def best_mcc(y, p):
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    P = ys.sum()
    N = len(ys) - P
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    mcc = (tp * N - fp * P) / np.sqrt((tp + fp) * (P) * (N) * (N + P - fp - tp) + 1e-12)
    k = int(np.argmax(mcc))
    return float(mcc[k]), float(ps[k])


def run_fold(fold, args, device):
    split = "train"
    p = pd.read_parquet(f"{OUT}/{split}_cand.parquet")
    foldcol = p.fold.values
    tr_idx = np.where(foldcol != fold)[0]
    va_idx = np.where(foldcol == fold)[0]
    if args.limit:
        rng = np.random.default_rng(0)
        tr_idx = rng.permutation(tr_idx)[:args.limit]
        va_idx = rng.permutation(va_idx)[:max(2000, args.limit // 8)]
    ds_tr = PairDS(split, tr_idx, True)
    ds_va = PairDS(split, va_idx, False)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    dl_va = DataLoader(ds_va, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)
    model = Net(args.backbone, ds_tr.geom.shape[1]).to(device)
    bb_params = list(model.bb.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr * 0.3},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.3, args.lr],
                                                total_steps=steps, pct_start=0.2)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    yva = p.contact.values[va_idx].astype(np.float32)
    best_auc = -1
    best_state = None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for it, (x, g, y) in enumerate(dl_tr):
            x = normalize(x.to(device, non_blocking=True))
            g = g.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                z = model(x, g)
                loss = crit(z.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if args.max_steps and it + 1 >= args.max_steps:
                break
        pva = predict(model, dl_va, device)
        auc = roc_auc_score(yva, pva)
        ap = average_precision_score(yva, pva)
        mcc, thr = best_mcc(yva, pva)
        print(f"fold{fold} ep{ep} loss={tot/max(1,it+1):.4f} auc={auc:.4f} ap={ap:.4f} mcc={mcc:.4f}@{thr:.3f} {time.time()-t0:.0f}s", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = pva
    return va_idx, pva, mcc, thr, model, best_state, best_pred, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    folds = list(range(5)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    print(args, flush=True)
    all_idx, all_p = [], []
    for fold in folds:
        idx, p, mcc, thr, model, best_state, best_pred, best_auc = run_fold(fold, args, device)
        all_idx.append(idx); all_p.append(p)
        if args.save:
            np.save(f"{OUT}/valpred_f{fold}_{args.tag}.npy", p)
            np.save(f"{OUT}/validx_f{fold}_{args.tag}.npy", idx)
            import json
            json.dump({"fold": fold, "args": vars(args), "best_mcc": mcc, "best_thr": thr},
                      open(f"{OUT}/meta_f{fold}_{args.tag}.json", "w"))
            torch.save(model.state_dict(), f"{OUT}/model_f{fold}_{args.tag}.pt")
            torch.save(best_state, f"{OUT}/model_f{fold}_{args.tag}_best.pt")
            best_auc_val = best_auc
    if args.save:
        idx = np.concatenate(all_idx); p = np.concatenate(all_p)
        pd.DataFrame({"row": idx, "pred": p}).to_parquet(f"{OUT}/oofpred_{args.tag}.parquet", index=False)


if __name__ == "__main__":
    main()
