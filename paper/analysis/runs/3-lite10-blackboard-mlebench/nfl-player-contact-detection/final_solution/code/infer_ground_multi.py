import os, sys, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn import Net, OUT
import infer_cnn


class GDS(Dataset):
    def __init__(self, split, idx, suffix, prefix="ground"):
        self.idx = np.asarray(idx, np.int64)
        self.arr = np.load(f"{OUT}/{split}_{prefix}_crops{suffix}.npy", mmap_mode="r")
        self.geom = np.load(f"{OUT}/geom_{prefix}_{split}.npy", mmap_mode="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = int(self.idx[i])
        x = np.ascontiguousarray(np.asarray(self.arr[r]).transpose(0, 3, 1, 2))
        return torch.from_numpy(x), torch.from_numpy(np.asarray(self.geom[r]).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--suffixes", default=",-6,6")
    ap.add_argument("--tagout", default="v1m")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefix", default="ground")
    args = ap.parse_args()
    suffixes = [s if (s == "" or s.startswith("_")) else f"_off{s}" for s in args.suffixes.split(",")]
    device = "cuda"
    tr_plan = pd.read_parquet(f"{OUT}/train_ground.parquet")
    te_plan = pd.read_parquet(f"{OUT}/test_ground.parquet")
    geomdim = np.load(f"{OUT}/geom_{args.prefix}_train.npy", mmap_mode="r").shape[1]
    oof = np.full(len(tr_plan), np.nan)
    for f in range(5):
        va = np.where(tr_plan.fold.values == f)[0]
        acc = np.zeros(len(va))
        for s in suffixes:
            dl = DataLoader(GDS("train", va, s, args.prefix), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
            wp = f"{OUT}/gmodel_f{f}_{args.tag}_best.pt"
            if not os.path.exists(wp):
                wp = f"{OUT}/gmodel_f{f}_{args.tag}.pt"
            model = Net("resnet18", geomdim).to(device)
            model.load_state_dict(torch.load(wp, map_location=device))
            acc += 0.5 * (infer_cnn.pred_model(model, dl, device, False) + infer_cnn.pred_model(model, dl, device, True))
            del model
        oof[va] = acc / len(suffixes)
        print("fold", f, "done", flush=True)
    op = tr_plan[["contact_id", "game_play", "step", "player", "contact", "fold"]].copy()
    op["pred"] = oof
    op.to_parquet(f"{OUT}/oof_ground_cnn_{args.tagout}.parquet", index=False)
    print("OOF saved", op.shape, "nan", int(np.isnan(oof).sum()), "mean", float(np.nanmean(oof)))
    # test
    ps = []
    idx = np.arange(len(te_plan))
    for s in suffixes:
        dl = DataLoader(GDS("test", idx, s, args.prefix), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        for f in range(5):
            wp = f"{OUT}/gmodel_f{f}_{args.tag}_best.pt"
            if not os.path.exists(wp):
                wp = f"{OUT}/gmodel_f{f}_{args.tag}.pt"
            model = Net("resnet18", geomdim).to(device)
            model.load_state_dict(torch.load(wp, map_location=device))
            ps.append(0.5 * (infer_cnn.pred_model(model, dl, device, False) + infer_cnn.pred_model(model, dl, device, True)))
            del model
        print("test array", s or "main", "done", flush=True)
    p = np.mean(ps, 0)
    pd.DataFrame({"contact_id": te_plan.contact_id.values, "pred": p}).to_parquet(
        f"{OUT}/test_ground_cnn_{args.tagout}.parquet", index=False)
    print("test saved", len(p), p.mean())


if __name__ == "__main__":
    main()
