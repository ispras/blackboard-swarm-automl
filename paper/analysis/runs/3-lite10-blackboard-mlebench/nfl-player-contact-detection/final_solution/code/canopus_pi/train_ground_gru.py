import os, sys, time, gc, argparse, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, matthews_corrcoef

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
NEWF = f"{ROOT}/workspaces/canopus_pi/features/new_feats_v2.parquet"
EXTRA = f"{ROOT}/workspaces/canopus_pi/features/ground_extra.parquet"
PRED = f"{ROOT}/workspaces/canopus_pi/preds"
sys.path.insert(0, f"{ROOT}/workspaces/canopus_pi")
from features_list import ground_features_v2
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

torch.manual_seed(42)
torch.set_num_threads(8)

EXTRA_FEATS = [
    "run_low0.5", "fut_low0.5", "run_low1.5", "fut_low1.5", "run_low2.5", "fut_low2.5",
    "speed_c5_min", "speed_c5_max", "speed_c5_mean",
    "sa_c5_min", "sa_c5_max", "sa_c5_mean",
    "acceleration_c5_min", "acceleration_c5_max", "acceleration_c5_mean",
    "dist_fut5_sum", "dist_past5_sum",
    "rank_speed", "rank_sa", "rank_acceleration", "rank_distance",
    "sid_top_rank2", "end_top_rank2", "sid_bottom_rank2", "end_bottom_rank2",
    "sid_h_rank2", "end_h_rank2",
]
FEATS = ground_features_v2() + EXTRA_FEATS


def read_cols(spec, nrows):
    total = sum(len(c) for _, c in spec)
    X = np.empty((nrows, total), dtype=np.float32)
    j = 0
    for path, cols in spec:
        pf = pq.ParquetFile(path)
        for c in cols:
            X[:, j] = pf.read(columns=[c]).column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            j += 1
    return X


class GRUNet(nn.Module):
    def __init__(self, F, h=96, layers=1):
        super().__init__()
        self.inp = nn.Linear(F, 128)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(128, h, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=(0.1 if layers > 1 else 0.0))
        self.head = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x, lengths):
        x = self.drop(self.act(self.inp(x)))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=x.size(1))
        return self.head(o).squeeze(-1)


def sweep(y, p):
    lo, hi = 0.4, 0.9998
    o = np.argsort(-p, kind="stable"); ys = y[o].astype(np.int64); ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = int(ys.sum()); N = len(y) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    qlo, qhi = np.quantile(p, [lo, hi]); mcc = np.where((ps >= qlo) & (ps <= qhi), mcc, -1)
    i = int(np.argmax(mcc)); return float(mcc[i]), float(ps[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--maxlen", type=int, default=180)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    meta = pd.read_parquet(FEAT, columns=["contact_id", "split", "contact", "is_ground", "step"])
    p = meta.contact_id.str.rsplit("_", n=3, expand=True)
    meta["gp"] = p[0].values
    meta["p1"] = p[2].astype(np.int64).values
    folds = pd.read_csv(f"{ROOT}/workspaces/alkaid_psi/cv_folds.csv")
    fmap = dict(zip(folds.game_play, folds.fold))
    meta["fold"] = meta["gp"].map(fmap).fillna(-1).astype(np.int16)
    meta["y"] = meta.contact.fillna(0).values.astype(np.float32)
    gidx = np.where((meta.is_ground == 1).values)[0]
    nfull = len(meta)
    meta = meta.iloc[gidx].reset_index(drop=True)
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    log("ground rows", len(meta))

    base_avail = set(pq.ParquetFile(FEAT).schema.names)
    new_avail = set(pq.ParquetFile(NEWF).schema.names)
    ex_avail = set(pq.ParquetFile(EXTRA).schema.names)
    fb = [c for c in FEATS if c in base_avail]
    fn = [c for c in FEATS if c in new_avail and c not in fb]
    fe = [c for c in FEATS if c in ex_avail and c not in fb and c not in fn]
    log("features", len(fb), len(fn), len(fe))
    X = read_cols([(FEAT, fb), (NEWF, fn), (EXTRA, fe)], nfull)
    X = X[gidx]
    # presence indicators
    def col_idx(name):
        for j, c in enumerate(FEATS):
            if c == name:
                return j
        return None
    pres = []
    for c in ["sid_h_top_1", "end_h_top_1"]:
        j = col_idx(c)
        pres.append((~np.isnan(X[:, j])).astype(np.float32) if j is not None else np.zeros(len(X), np.float32))
    X = np.column_stack([X] + pres).astype(np.float32)
    log("X", X.shape, X.nbytes / 1e9)
    del base_avail, new_avail, ex_avail
    gc.collect()

    # sort and build sequences
    meta = meta.sort_values(["gp", "p1", "step"]).reset_index(drop=True)
    lens, seq_groups = [], []
    for (gp, pl), idx in meta.groupby(["gp", "p1"], sort=False).indices.items():
        idx = np.sort(idx)
        if len(idx) > args.maxlen:
            idx = idx[:args.maxlen]
        seq_groups.append(idx)
        lens.append(len(idx))
    lens = np.array(lens, dtype=np.int64)
    nseq = len(seq_groups)
    T = int(lens.max())
    F = X.shape[1]
    log("nseq", nseq, "maxlen", T, "F", F)
    Xseq = np.zeros((nseq, T, F), dtype=np.float32)
    yseq = np.zeros((nseq, T), dtype=np.float32)
    rows_sorted = meta["row"].values
    y_sorted = meta["y"].values
    for gi, idx in enumerate(seq_groups):
        rr = rows_sorted[idx]
        L = len(idx)
        Xseq[gi, :L] = X[rr]
        yseq[gi, :L] = y_sorted[idx]
    seq_fold = meta.groupby(["gp", "p1"], sort=False)["fold"].first().values.astype(np.int16)
    seq_split = meta.groupby(["gp", "p1"], sort=False)["split"].first().values.astype(np.int8)
    del X
    gc.collect()

    # standardize on train
    trmask = seq_split == 0
    mu = np.nanmean(Xseq[trmask], axis=(0, 1))
    sd = np.nanstd(Xseq[trmask], axis=(0, 1)) + 1e-6
    rs = np.ones(F, dtype=bool)
    # don't standardize indicator cols
    mu[-2:] = 0.0; sd[-2:] = 1.0
    Xseq = (Xseq - mu) / sd
    np.nan_to_num(Xseq, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(Xseq, -10, 10, out=Xseq)
    log("standardized")
    Xt = torch.from_numpy(Xseq)
    yt = torch.from_numpy(yseq)
    lens_t = torch.from_numpy(lens)

    oof_rows = np.full(len(meta), np.nan, dtype=np.float32)
    test_rows = np.full(len(meta), np.nan, dtype=np.float32)
    test_sum_rows = np.zeros(len(meta), dtype=np.float32)
    val_scores = []

    for k in range(5):
        trg = np.where((seq_fold != k) & (seq_fold >= 0))[0]
        vag = np.where(seq_fold == k)[0]
        teg = np.where(seq_split == 1)[0]
        net = GRUNet(F, args.hidden, args.layers)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(2.5))
        best_auc = -1; best_state = None; bad = 0
        for ep in range(args.epochs):
            net.train()
            perm = np.random.RandomState(1000 + k * 100 + ep).permutation(len(trg))
            tot = 0.0; nb = 0
            for bi in range(0, len(trg), 64):
                sel = trg[perm[bi:bi + 64]]
                xb = Xt[sel]; lb = lens_t[sel]; yb = yt[sel]
                logits = net(xb, lb)
                mask = torch.arange(xb.size(1))[None, :] < lb[:, None]
                loss = (lossf(logits, yb) * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()
                tot += float(loss); nb += 1
            # validate
            net.eval()
            vals = []
            with torch.no_grad():
                for bi in range(0, len(vag), 256):
                    sel = vag[bi:bi + 256]
                    logits = net(Xt[sel], lens_t[sel])
                    out = torch.sigmoid(logits)
                    for j, gi in enumerate(sel):
                        vals.append(out[j, :lens[gi]].numpy())
            vp = np.concatenate(vals)
            vy = np.concatenate([yseq[gi, :lens[gi]] for gi in vag])
            auc = roc_auc_score(vy, vp)
            log(f"fold{k} ep{ep} loss {tot/max(nb,1):.4f} val_auc {auc:.5f}")
            if auc > best_auc + 1e-5:
                best_auc = auc; best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        # oof
        with torch.no_grad():
            for bi in range(0, len(vag), 256):
                sel = vag[bi:bi + 256]
                out = torch.sigmoid(net(Xt[sel], lens_t[sel]))
                for j, gi in enumerate(sel):
                    rr = rows_sorted[seq_groups[gi]]
                    oof_rows[rr] = out[j, :lens[gi]].numpy()
        # test
        with torch.no_grad():
            for bi in range(0, len(teg), 256):
                sel = teg[bi:bi + 256]
                out = torch.sigmoid(net(Xt[sel], lens_t[sel]))
                for j, gi in enumerate(sel):
                    rr = rows_sorted[seq_groups[gi]]
                    test_rows[rr] = out[j, :lens[gi]].numpy()
        test_sum_rows += test_rows
        log(f"fold{k} best_auc {best_auc:.5f}")
        val_scores.append(best_auc)
        del net, opt, best_state
        gc.collect()

    # restore original meta order and save
    meta2 = meta.sort_values("row").reset_index(drop=True)
    oof = oof_rows[meta2["row"].values]
    tr_sel = (meta2.split == 0).values
    y = meta2.y.values[tr_sel].astype(np.int8)
    pv = oof[tr_sel]
    mcc, th = sweep(y, pv)
    auc = roc_auc_score(y, pv)
    log(f"GROUND GRU OOF MCC {mcc:.5f} th {th:.4f} AUC {auc:.5f}")
    pd.DataFrame({"contact_id": meta2.contact_id.values[tr_sel], "pred": pv,
                  "fold": meta2.fold.values[tr_sel]}).to_parquet(
        f"{PRED}/oof_ground_gru_{args.tag}.parquet", index=False)
    te_sel = (meta2.split == 1).values
    test_out = (test_sum_rows / 5.0)[meta2["row"].values]
    pd.DataFrame({"contact_id": meta2.contact_id.values[te_sel],
                  "pred": test_out[te_sel]}).to_parquet(
        f"{PRED}/test_ground_gru_{args.tag}.parquet", index=False)
    json.dump({"mcc": mcc, "th": th, "auc": auc, "val_aucs": val_scores},
              open(f"{PRED}/metrics_ground_gru_{args.tag}.json", "w"), indent=2)
    log("saved")


if __name__ == "__main__":
    main()
