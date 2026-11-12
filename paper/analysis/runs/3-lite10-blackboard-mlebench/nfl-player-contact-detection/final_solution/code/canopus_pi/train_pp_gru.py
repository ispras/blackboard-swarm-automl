import os, sys, time, gc, argparse, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
NEWF = f"{ROOT}/workspaces/canopus_pi/features/new_feats_v2.parquet"
CNN = f"{ROOT}/workspaces/canopus_pi/features/pp_cnn_stack.parquet"
PRED = f"{ROOT}/workspaces/canopus_pi/preds"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(42)
torch.set_num_threads(8)

PAIR = ["dist", "dx_field", "dy_field", "closing_speed", "rel_speed", "speed_sum", "speed_diff",
        "same_team", "dir_diff", "ori_diff", "dir1_to_line", "dir2_to_line",
        "dist_lag1", "dist_lead1", "dist_d1", "dist_fd1", "dist_roll5_min", "dist_fut5_min",
        "log_dist", "inv_dist", "time_to_collision",
        "pair_third_min", "pair_c2_1", "pair_c2_2", "pair_is_nn1", "pair_is_nn2",
        "nb1_1", "nb1_2", "n2_1", "n2_2", "nn_mutual_1", "nn_mutual_2",
        "sid_pair_dpx", "end_pair_dpx", "pair_dpx_min", "pair_dpx_mean",
        "sid_pair_iou", "end_pair_iou", "sid_pair_dpx_norm", "end_pair_dpx_norm",
        "pdpx_lag1", "pdpx_lead1", "pdpx_win5_min", "views_both"]
PLAYER = ["speed", "acceleration", "sa", "x_position", "y_position", "speed_d1",
          "sa_d1", "speed_roll5_mean", "sa_roll5_min", "ori_change1", "position_code", "team_home"]
HELMET = ["h_top", "h_cy", "h_h", "top_rank"]
FEATS = PAIR[:]
for c in PLAYER:
    FEATS += [f"{c}_1", f"{c}_2"]
for view in ["sid", "end"]:
    for c in HELMET:
        FEATS += [f"{view}_{c}_1", f"{view}_{c}_2"]
FEATS += ["cnn_v2m", "cnn_v2tta"]


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
        self.gru = nn.GRU(128, h, layers, batch_first=True, bidirectional=True,
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
    qlo, qhi = np.quantile(p, [lo, hi]); mcc = np.where((ps >= qlo) & (ps <= hi), mcc, -1)
    i = int(np.argmax(mcc)); return float(mcc[i]), float(ps[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--neg_frac", type=float, default=0.30)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    meta = pd.read_parquet(FEAT, columns=["contact_id", "split", "contact", "is_ground", "step"])
    p = meta.contact_id.str.rsplit("_", n=3, expand=True)
    meta["gp"] = p[0].values
    meta["p1"] = p[2].astype(np.int64).values
    meta["p2n"] = pd.to_numeric(p[3], errors="coerce").fillna(-1).astype(np.int64).values
    folds = pd.read_csv(f"{ROOT}/workspaces/alkaid_psi/cv_folds.csv")
    fmap = dict(zip(folds.game_play, folds.fold))
    meta["fold"] = meta["gp"].map(fmap).fillna(-1).astype(np.int16)
    meta["y"] = meta.contact.fillna(0).values.astype(np.float32)
    idx = np.where((meta.is_ground == 0).values)[0]
    nfull = len(meta)
    meta = meta.iloc[idx].reset_index(drop=True)
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    log("pp rows", len(meta))

    b_av = set(pq.ParquetFile(FEAT).schema.names)
    n_av = set(pq.ParquetFile(NEWF).schema.names)
    c_av = set(pq.ParquetFile(CNN).schema.names)
    fb = [c for c in FEATS if c in b_av]
    fn = [c for c in FEATS if c in n_av and c not in fb]
    fc = [c for c in FEATS if c in c_av and c not in fb and c not in fn]
    log("feats", len(fb), len(fn), len(fc))
    X = read_cols([(FEAT, fb), (NEWF, fn), (CNN, fc)], nfull)[idx]
    # missing indicators
    for name in ["cnn_v2m", "sid_pair_dpx", "end_pair_dpx"]:
        if name in FEATS:
            j = FEATS.index(name)
            X = np.column_stack([X, (~np.isnan(X[:, j])).astype(np.float32)]).astype(np.float32)
    F = X.shape[1]
    log("X", X.shape, X.nbytes / 1e9)

    meta = meta.sort_values(["gp", "p1", "p2n", "step"]).reset_index(drop=True)
    groups = []
    for _, ii in meta.groupby(["gp", "p1", "p2n"], sort=False).indices.items():
        ii = np.sort(ii)
        groups.append(ii)
    lens = np.array([len(g) for g in groups], dtype=np.int64)
    nseq = len(groups)
    rows_sorted = meta["row"].values
    seq_rows = [rows_sorted[g] for g in groups]
    seq_y = [meta["y"].values[g] for g in groups]
    seq_fold = meta.groupby(["gp", "p1", "p2n"], sort=False)["fold"].first().values.astype(np.int16)
    seq_split = meta.groupby(["gp", "p1", "p2n"], sort=False)["split"].first().values.astype(np.int8)
    seq_pos = np.array([y.sum() > 0 for y in seq_y])
    log("nseq", nseq, "pos seq", seq_pos.sum())

    train_rows = meta["row"].values[meta.split.values == 0]
    mu = np.nanmean(X[train_rows], axis=0)
    sd = np.nanstd(X[train_rows], axis=0) + 1e-6
    for j in range(F - 3, F):
        mu[j] = 0.0; sd[j] = 1.0
    X -= mu; X /= sd
    np.nan_to_num(X, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(X, -10, 10, out=X)
    log("standardized")
    Xt = torch.from_numpy(X)

    oof_rows = np.full(len(meta), np.nan, dtype=np.float32)
    test_sum_rows = np.zeros(len(meta), dtype=np.float32)
    val_scores = []
    for k in range(5):
        trg = np.where((seq_fold != k) & (seq_fold >= 0))[0]
        vag = np.where(seq_fold == k)[0]
        teg = np.where(seq_split == 1)[0]
        rng = np.random.RandomState(100 + k)
        pos_s = trg[seq_pos[trg]]
        neg_s = trg[~seq_pos[trg]]
        neg_s = neg_s[rng.rand(len(neg_s)) < args.neg_frac]
        tr_s = np.concatenate([pos_s, neg_s]); rng.shuffle(tr_s)
        log(f"fold{k} train seq {len(tr_s)} (pos {len(pos_s)}) val {len(vag)}")
        net = GRUNet(F, args.hidden, args.layers)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(3.0))
        best_auc = -1; best_state = None; bad = 0

        def build_batch(sel):
            ml = int(lens[sel].max())
            B = len(sel)
            xb = np.zeros((B, ml, F), dtype=np.float32)
            yb = np.zeros((B, ml), dtype=np.float32)
            lb = np.zeros(B, dtype=np.int64)
            for j, gi in enumerate(sel):
                rr = seq_rows[gi]; L = len(rr)
                xb[j, :L] = X[rr]; yb[j, :L] = seq_y[gi]; lb[j] = L
            return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(lb)

        for ep in range(args.epochs):
            net.train()
            perm = np.random.RandomState(2000 + k * 100 + ep).permutation(len(tr_s))
            tot = 0.0; nb = 0
            for bi in range(0, len(tr_s), 32):
                sel = tr_s[perm[bi:bi + 32]]
                xb, yb, lb = build_batch(sel)
                logits = net(xb, lb)
                mask = torch.arange(xb.size(1))[None, :] < lb[:, None]
                loss = (lossf(logits, yb) * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()
                tot += float(loss); nb += 1
            net.eval()
            vp = []
            with torch.no_grad():
                for bi in range(0, len(vag), 256):
                    sel = vag[bi:bi + 256]
                    xb, yb, lb = build_batch(sel)
                    out = torch.sigmoid(net(xb, lb))
                    for j, gi in enumerate(sel):
                        vp.append(out[j, :lens[gi]].numpy())
            vp = np.concatenate(vp)
            vy = np.concatenate([seq_y[gi] for gi in vag])
            auc = roc_auc_score(vy, vp)
            log(f"fold{k} ep{ep} loss {tot/max(nb,1):.4f} val_auc {auc:.5f}")
            if auc > best_auc + 1e-5:
                best_auc = auc; best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            for bi in range(0, len(vag), 256):
                sel = vag[bi:bi + 256]
                xb, _, lb = build_batch(sel)
                out = torch.sigmoid(net(xb, lb))
                for j, gi in enumerate(sel):
                    oof_rows[seq_rows[gi]] = out[j, :lens[gi]].numpy()
            test_rows = np.zeros(len(meta), dtype=np.float32)
            for bi in range(0, len(teg), 256):
                sel = teg[bi:bi + 256]
                xb, _, lb = build_batch(sel)
                out = torch.sigmoid(net(xb, lb))
                for j, gi in enumerate(sel):
                    test_rows[seq_rows[gi]] = out[j, :lens[gi]].numpy()
        test_sum_rows += test_rows
        val_scores.append(best_auc)
        log(f"fold{k} best_auc {best_auc:.5f}")
        del net, opt, best_state; gc.collect()

    meta2 = meta.sort_values("row").reset_index(drop=True)
    tr_sel = meta2.split.values == 0
    y = meta2.y.values[tr_sel].astype(np.int8)
    pv = oof_rows[meta2["row"].values][tr_sel]
    mcc, th = sweep(y, pv)
    auc = roc_auc_score(y, pv)
    log(f"PP GRU OOF MCC {mcc:.5f} th {th:.4f} AUC {auc:.5f}")
    pd.DataFrame({"contact_id": meta2.contact_id.values[tr_sel], "pred": pv,
                  "fold": meta2.fold.values[tr_sel]}).to_parquet(
        f"{PRED}/oof_pp_gru_{args.tag}.parquet", index=False)
    te_sel = meta2.split.values == 1
    test_out = (test_sum_rows / 5.0)[meta2["row"].values]
    pd.DataFrame({"contact_id": meta2.contact_id.values[te_sel],
                  "pred": test_out[te_sel]}).to_parquet(
        f"{PRED}/test_pp_gru_{args.tag}.parquet", index=False)
    json.dump({"mcc": mcc, "th": th, "auc": auc, "val_aucs": val_scores},
              open(f"{PRED}/metrics_pp_gru_{args.tag}.json", "w"), indent=2)
    log("saved")


if __name__ == "__main__":
    main()
