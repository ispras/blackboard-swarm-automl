import os, sys, time, argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn import Net, normalize
from train_ground import best_mcc, predict
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "workspaces/alkaid_psi"


class DS(Dataset):
    def __init__(self, split, idx, train, prefix):
        self.idx = np.asarray(idx, np.int64)
        self.train = train
        base = f"{split}_crops" if not prefix else f"{split}_{prefix}_crops"
        gname = split if not prefix else f"{prefix}_{split}"
        self.arr = np.load(f"{OUT}/{base}.npy", mmap_mode="r")
        self.geom = np.load(f"{OUT}/geom_{gname}.npy", mmap_mode="r")
        p = pd.read_parquet(f"{OUT}/{split}_cand.parquet")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--prefix", default="ppW")
    ap.add_argument("--tag", default="vW1")
    ap.add_argument("--backbone", default="resnet18")
    args = ap.parse_args()
    device = "cuda"
    plan = pd.read_parquet(f"{OUT}/train_cand.parquet")
    gname = "train" if not args.prefix else f"{args.prefix}_train"
    geomdim = np.load(f"{OUT}/geom_{gname}.npy", mmap_mode="r").shape[1]
    for fold in range(5):
        tr_idx = np.where(plan.fold.values != fold)[0]
        va_idx = np.where(plan.fold.values == fold)[0]
        ds_tr = DS("train", tr_idx, True, args.prefix); ds_va = DS("train", va_idx, False, args.prefix)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                           drop_last=True, persistent_workers=args.workers > 0)
        dl_va = DataLoader(ds_va, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
        model = Net(args.backbone, geomdim).to(device)
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
            print(f"wfold{fold} ep{ep} loss={tot/max(1,it+1):.4f} auc={auc:.4f} ap={ap:.4f} mcc={mcc:.4f}@{thr:.3f} {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        np.save(f"{OUT}/valpred_f{fold}_{args.tag}.npy", pva)
        np.save(f"{OUT}/validx_f{fold}_{args.tag}.npy", va_idx)
        torch.save(model.state_dict(), f"{OUT}/model_f{fold}_{args.tag}.pt")
        torch.save(best_state, f"{OUT}/model_f{fold}_{args.tag}_best.pt")
        json.dump({"fold": fold, "auc": best_auc, "args": vars(args)}, open(f"{OUT}/meta_f{fold}_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
