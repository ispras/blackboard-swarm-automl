import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import FEATS_V2, TE_COLS, parts_v2, load_te

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
CONT = FEATS_V2 + TE_COLS

CAT_SPECS = [("year", 8), ("month", 13), ("weekday", 7), ("hour", 24),
             ("pax", 9), ("pcell", 11820), ("dcell", 12765)]


class Net(nn.Module):
    def __init__(self, ncont, emb_dims, hidden=(512, 256, 128), pdrop=0.05):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d) for (v, d) in emb_dims])
        emb_total = sum(d for (_, d) in emb_dims)
        d_in = ncont + emb_total
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(pdrop)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xc, cats):
        em = [e(c) for e, c in zip(self.embs, cats)]
        x = torch.cat([xc] + em, dim=1)
        return self.mlp(x).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tag", default="nn1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxparts", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    t0 = time.time()

    te = load_te()
    up = te["u_pcell"]; ud = te["u_dcell"]
    pl = parts_v2()
    if args.maxparts > 0:
        pl = pl[:args.maxparts]
    XL, YL, OIL, CL, TL = [], [], [], [], []
    off = 0
    for p in pl:
        df = pd.read_parquet(p, columns=["orig_idx", "fare_amount"] + FEATS_V2 + ["p_cell", "d_cell"])
        n = len(df)
        XL.append(df[FEATS_V2].to_numpy(np.float32))
        TL.append(np.stack([te[c][off:off + n] for c in TE_COLS], axis=1))
        YL.append(df["fare_amount"].to_numpy(np.float32))
        OIL.append(df["orig_idx"].to_numpy(np.int64))
        pc = np.clip(np.searchsorted(up, df["p_cell"].to_numpy(np.int64)), 0, len(up) - 1).astype(np.int32)
        dc = np.clip(np.searchsorted(ud, df["d_cell"].to_numpy(np.int64)), 0, len(ud) - 1).astype(np.int32)
        CL.append(np.stack([df["year"].to_numpy(np.float32).astype(np.int32) - 2009,
                            df["month"].to_numpy(np.float32).astype(np.int32) - 1,
                            df["weekday"].to_numpy(np.float32).astype(np.int32),
                            df["hour"].to_numpy(np.float32).astype(np.int32),
                            np.clip(df["passenger_count"].to_numpy(np.float32).astype(np.int32), 0, 8),
                            pc, dc], axis=1))
        del df
        print("loaded part", off, "->", off + n, time.time() - t0, flush=True)
        off += n
    X = np.concatenate(XL); Y = np.concatenate(YL); OI = np.concatenate(OIL); C = np.concatenate(CL)
    X = np.hstack([X, np.concatenate(TL).astype(np.float32)])
    del XL, YL, OIL, CL, TL
    n = len(Y)
    print("all loaded", X.shape, C.shape, time.time() - t0, flush=True)

    hold = (OI % 100 == 0)
    tr = np.where(~hold)[0]; va = np.where(hold)[0]
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    X -= mu; X /= sd
    ym = float(Y.mean()); ys = float(Y.std())
    Xg = torch.from_numpy(X).to(dev)
    Cg = torch.from_numpy(C).long().to(dev)
    Yg = torch.from_numpy((Y - ym) / ys).to(dev)
    trg = torch.from_numpy(tr.astype(np.int64)).to(dev)
    vag = torch.from_numpy(va.astype(np.int64)).to(dev)
    del X, C, Y
    print("moved to gpu", time.time() - t0, flush=True)

    emb_dims = [((CAT_SPECS[i][1]), [4, 4, 4, 8, 4, 16, 16][i]) for i in range(len(CAT_SPECS))]
    model = Net(len(CONT), emb_dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ntr = len(trg)
    steps_per = (ntr + args.bs - 1) // args.bs
    total = steps_per * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.1, anneal_strategy="cos")
    lossf = nn.MSELoss()

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        preds = torch.empty(len(idx), device=dev)
        for i in range(0, len(idx), 65536):
            b = idx[i:i + 65536]
            preds[i:i + len(b)] = model(Xg[b], [Cg[b, j] for j in range(Cg.shape[1])])
        preds = preds * ys + ym
        yt = Yg[idx] * ys + ym
        return torch.sqrt(torch.mean((preds - yt) ** 2)).item(), preds.cpu().numpy()

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        perm = trg[torch.randperm(ntr, device=dev)]
        run = 0.0
        for bi, i in enumerate(range(0, ntr, args.bs)):
            b = perm[i:i + args.bs]
            xb = Xg[b]
            yb = Yg[b]
            cats = [Cg[b, j] for j in range(Cg.shape[1])]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(xb, cats)
                loss = lossf(out.float(), yb)
            loss.backward()
            opt.step(); sched.step()
            run += loss.item()
            if bi % 500 == 0:
                print(f"ep{ep} step{bi}/{steps_per} loss {run/(bi+1):.5f} t={time.time()-t0:.0f}", flush=True)
        rm, pv = evaluate(vag)
        print(f"EPOCH {ep} VAL_RMSE {rm:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if rm < best:
            best = rm
            torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "ym": ym, "ys": ys,
                        "emb_dims": emb_dims}, os.path.join(OUT, f"nn_{args.tag}.pt"))
            np.save(os.path.join(OUT, f"holdout_pred_{args.tag}.npy"), pv.astype(np.float32))
            np.save(os.path.join(OUT, f"holdout_orig_{args.tag}.npy"), OI[va].astype(np.int64))
            np.save(os.path.join(OUT, f"holdout_true_{args.tag}.npy"), (Yg[vag] * ys + ym).cpu().numpy().astype(np.float32))
    json.dump({"best_val_rmse": best, "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
               "time_s": time.time() - t0}, open(os.path.join(OUT, f"result_{args.tag}.json"), "w"), indent=2)
    print("BEST", best, "done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
