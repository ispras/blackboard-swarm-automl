import os, sys, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn import Net, OUT
import infer_cnn


class MultiDS(Dataset):
    def __init__(self, split, idx, suffix, prefix=""):
        self.idx = np.asarray(idx, np.int64)
        base = f"{split}_crops" if not prefix else f"{split}_{prefix}_crops"
        self.arr = np.load(f"{OUT}/{base}{suffix}.npy", mmap_mode="r")
        gname = split if not prefix else f"{prefix}_{split}"
        self.geom = np.load(f"{OUT}/geom_{gname}.npy", mmap_mode="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = int(self.idx[i])
        x = np.ascontiguousarray(np.asarray(self.arr[r]).transpose(0, 3, 1, 2))
        return torch.from_numpy(x), torch.from_numpy(np.asarray(self.geom[r]).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--suffixes", default=",-6,6")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tagout", default="v2m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nviews", type=int, default=2)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--backbone", default="resnet18")
    args = ap.parse_args()
    suffixes = [s if (s == "" or s.startswith("_")) else f"_off{s}" for s in args.suffixes.split(",")]
    device = "cuda"
    plan = pd.read_parquet(f"{OUT}/{'train_cand' if args.split == 'train' else 'test_cand'}.parquet")
    gname = args.split if not args.prefix else f"{args.prefix}_{args.split}"
    geomdim = np.load(f"{OUT}/geom_{gname}.npy", mmap_mode="r").shape[1]
    folds = list(range(5))
    fold_p = {f: [] for f in folds}
    if args.split == "train":
        for f in folds:
            idx = np.where(plan.fold.values == f)[0]
            for s in suffixes:
                dl = DataLoader(MultiDS("train", idx, s, args.prefix), batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
                wp = f"{OUT}/model_f{f}_{args.tag}_best.pt"
                if not os.path.exists(wp):
                    wp = f"{OUT}/model_f{f}_{args.tag}.pt"
                model = Net(args.backbone, geomdim, nviews=args.nviews).to(device)
                model.load_state_dict(torch.load(wp, map_location=device))
                p = 0.5 * (infer_cnn.pred_model(model, dl, device, False) + infer_cnn.pred_model(model, dl, device, True))
                fold_p[f].append(p)
                del model
            print("fold", f, "done", flush=True)
    else:
        idx = np.arange(len(plan))
        for s in suffixes:
            dl = DataLoader(MultiDS("test", idx, s, args.prefix), batch_size=args.bs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
            for f in folds:
                wp = f"{OUT}/model_f{f}_{args.tag}_best.pt"
                if not os.path.exists(wp):
                    wp = f"{OUT}/model_f{f}_{args.tag}.pt"
                model = Net(args.backbone, geomdim, nviews=args.nviews).to(device)
                model.load_state_dict(torch.load(wp, map_location=device))
                p = 0.5 * (infer_cnn.pred_model(model, dl, device, False) + infer_cnn.pred_model(model, dl, device, True))
                fold_p[f].append(p)
                del model
            print("array", s or "main", "done", flush=True)
    if args.split == "test":
        p = np.mean([np.mean(fold_p[f], 0) for f in folds], 0)
        out = args.out or f"{OUT}/test_cnn_cand_{args.tagout}.parquet"
        pd.DataFrame({"contact_id": plan.contact_id.values, "pred": p}).to_parquet(out, index=False)
        print("saved", out, p.mean())
    else:
        for f in folds:
            va = np.where(plan.fold.values == f)[0]
            p = np.mean(fold_p[f], 0)
            np.save(f"{OUT}/valpred_f{f}_{args.tagout}.npy", p)
            np.save(f"{OUT}/validx_f{f}_{args.tagout}.npy", va)
        print("saved train val preds", args.tagout)


if __name__ == "__main__":
    main()
