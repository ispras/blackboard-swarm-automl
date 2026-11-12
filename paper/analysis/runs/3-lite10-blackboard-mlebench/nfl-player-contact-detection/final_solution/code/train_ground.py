import os, sys, time, argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn import Net, normalize, MEAN, STD
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "workspaces/alkaid_psi"


class GroundDS(Dataset):
    def __init__(self, split, idx, train, prefix="ground"):
        self.idx = np.asarray(idx, np.int64)
        self.train = train
        self.arr = np.load(f"{OUT}/{split}_{prefix}_crops.npy", mmap_mode="r")
        self.geom = np.load(f"{OUT}/geom_{prefix}_{split}.npy", mmap_mode="r")
        p = pd.read_parquet(f"{OUT}/{split}_ground.parquet")
        self.y = p.contact.values.astype(np.float32) if "contact" in p else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = int(self.idx[i])
        x = np.asarray(self.arr[r])
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                f = np.float32(np.random.uniform(0.85, 1.15)); b = np.float32(np.random.uniform(-15, 15))
                x = np.clip(x.astype(np.float32) * f + b, 0, 255).astype(np.uint8)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        return (torch.from_numpy(x), torch.from_numpy(np.asarray(self.geom[r]).copy()),
                torch.tensor(self.y[r] if self.y is not None else 0.0))


def best_mcc(y, p):
    o = np.argsort(-p); ys = y[o]; ps = p[o]; P = ys.sum(); N = len(ys) - P
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    den = np.sqrt((tp + fp) * P * N * (N + P - fp - tp))
    mcc = np.where(den > 0, (tp * N - fp * P) / (den + 1e-12), -1)
    k = int(np.argmax(mcc)); return float(mcc[k]), float(ps[k])


@torch.no_grad()
def predict(model, dl, device, flip=False):
    model.eval(); out = []
    for x, g, y in dl:
        x = x.to(device, non_blocking=True)
        if flip:
            x = x.flip(-1)
        x = normalize(x); g = g.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            z = model(x, g)
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(out)


class PGroundDS(GroundDS):
    def __getitem__(self, i):
        x, g, y = super().__getitem__(i)
        return x, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--neg_frac", type=float, default=1.0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--prefix", default="ground")
    args = ap.parse_args()
    device = "cuda"
    plan = pd.read_parquet(f"{OUT}/train_ground.parquet")
    foldcol = plan.fold.values
    for fold in range(5):
        tr_all = np.where(foldcol != fold)[0]
        va_idx = np.where(foldcol == fold)[0]
        pos = tr_all[plan.contact.values[tr_all] == 1]
        neg = tr_all[plan.contact.values[tr_all] == 0]
        if args.neg_frac < 1:
            rng = np.random.default_rng(0)
            neg = rng.choice(neg, int(len(pos) / max(1e-6, plan.contact.values[tr_all].mean()) * args.neg_frac / (1 - plan.contact.values[tr_all].mean())), replace=False) if False else rng.choice(neg, int(len(pos) * args.neg_frac), replace=False)
        tr_idx = np.concatenate([pos, neg])
        ds_tr = GroundDS("train", tr_idx, True, args.prefix); ds_va = GroundDS("train", va_idx, False, args.prefix)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                           drop_last=True, persistent_workers=args.workers > 0)
        dl_va = DataLoader(ds_va, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
        model = Net("resnet18", ds_tr.geom.shape[1]).to(device)
        opt = torch.optim.AdamW([{"params": list(model.bb.parameters()), "lr": args.lr * 0.3},
                                 {"params": list(model.head.parameters()), "lr": args.lr}], weight_decay=1e-4)
        steps = len(dl_tr) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.3, args.lr], total_steps=steps, pct_start=0.25)
        scaler = torch.cuda.amp.GradScaler()
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
        yva = plan.contact.values[va_idx].astype(np.float32)
        best_auc = -1; best_state = None
        for ep in range(args.epochs):
            model.train(); t0 = time.time(); tot = 0
            for it, (x, g, y) in enumerate(dl_tr):
                x = normalize(x.to(device, non_blocking=True)); g = g.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss = crit(model(x, g).float(), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item()
            pva = predict(model, dl_va, device)
            auc = roc_auc_score(yva, pva); ap = average_precision_score(yva, pva)
            mcc, thr = best_mcc(yva, pva)
            print(f"gfold{fold} ep{ep} loss={tot/max(1,it+1):.4f} auc={auc:.4f} ap={ap:.4f} mcc={mcc:.4f}@{thr:.3f} {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        np.save(f"{OUT}/gvalpred_f{fold}_{args.tag}.npy", pva)
        np.save(f"{OUT}/gvalidx_f{fold}_{args.tag}.npy", va_idx)
        torch.save(model.state_dict(), f"{OUT}/gmodel_f{fold}_{args.tag}.pt")
        torch.save(best_state, f"{OUT}/gmodel_f{fold}_{args.tag}_best.pt")
        json.dump({"fold": fold, "auc": best_auc, "args": vars(args)},
                  open(f"{OUT}/gmeta_f{fold}_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
