import os, sys, json, argparse, itertools
import numpy as np, pandas as pd

WS = "/download_job_code/payload/sandbox/workspaces/capella_zeta"
S = 320
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def id_order(ids_csv):
    ids = pd.read_csv(ids_csv)
    id2pos = {i: k for k, i in enumerate(ids["id"])}
    return ids, id2pos


def load_specs(specs):
    vspecs, tspecs = [], []
    for s in specs:
        vi, vmap = id_order(s["val_ids"])
        ti, tmap = id_order(s["test_ids"])
        vspecs.append({"name": s["name"], "probs": np.load(s["val_probs"], mmap_mode="r"),
                       "ids": vi, "map": vmap, "w": s.get("w", 1.0)})
        tspecs.append({"name": s["name"], "probs": np.load(s["test_probs"], mmap_mode="r"),
                       "ids": ti, "map": tmap, "w": s.get("w", 1.0)})
    return vspecs, tspecs


def combine(specs, ref_ids_csv, out_path, weights, chunk=256):
    ref, _ = id_order(ref_ids_csv)
    N = len(ref)
    orders = [np.array([sp["map"][i] for i in ref["id"]]) for sp in specs]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, 3, S, S))
    tot = sum(weights)
    for st in range(0, N, chunk):
        en = min(st + chunk, N)
        acc = None
        for sp, od, w in zip(specs, orders, weights):
            a = np.asarray(sp["probs"][od[st:en]], dtype=np.float32) * (w / tot)
            acc = a if acc is None else acc + a
        out[st:en] = acc.astype(np.float16)
    out.flush()
    print("saved", out_path, N)
    return ref


def eval_weights(specs, ref, gt_file, weight_list, thrs=(0.45, 0.55, 0.65), chunk=256):
    idx = pd.read_csv(f"{WS}/cache/train_index.csv")
    id2row = dict(zip(idx["id"], np.arange(len(idx))))
    rows = np.array([id2row[i] for i in ref["id"]])
    gt = np.load(gt_file, mmap_mode="r")
    N = len(ref)
    orders = [np.array([sp["map"][i] for i in ref["id"]]) for sp in specs]
    results = []
    for wi, w in enumerate(weight_list):
        tot = sum(w)
        acc = {t: np.zeros(3) for t in thrs}
        for st in range(0, N, chunk):
            en = min(st + chunk, N)
            p = None
            for sp, od, ww in zip(specs, orders, w):
                a = np.asarray(sp["probs"][od[st:en]], dtype=np.float32) * (ww / tot)
                p = a if p is None else p + a
            g = np.stack([gt[rows[st:en], c] for c in range(3)], 1).astype(bool)
            for t in thrs:
                pr = p > t
                for c in range(3):
                    pc = pr[:, c]; gc = g[:, c]
                    inter = (pc & gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
                    d = np.where(den > 0, 2 * inter / np.maximum(den, 1), 1.0)
                    acc[t][c] += d.sum()
        score = {t: (acc[t] / N) for t in thrs}
        mean = np.mean([score[t].mean() for t in thrs])
        results.append((mean, w, {t: score[t].round(4).tolist() for t in thrs}))
        print(f"w={w} mean={mean:.4f}", {t: score[t].mean().round(4) for t in thrs}, flush=True)
    results.sort(key=lambda x: -x[0])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--mode", default="eval")  # eval | combine_val | combine_test
    ap.add_argument("--weights", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    specs = json.loads(a.specs)
    vspecs, tspecs = load_specs(specs)
    print("model val shapes:", [(s["name"], s["probs"].shape) for s in vspecs])
    print("model test shapes:", [(s["name"], s["probs"].shape) for s in tspecs])
    if a.mode == "eval":
        ref, _ = id_order(a.out or specs[0]["val_ids"])
        wl = json.loads(a.weights) if a.weights else \
            [[w1, w2, w3] for w1 in (0.5, 1, 1.5) for w2 in (0.5, 1, 1.5) for w3 in (0.5, 1, 1.5)]
        res = eval_weights(vspecs, ref, f"{WS}/cache/train_mask{S}.npy", wl)
        print("BEST", res[:3])
    elif a.mode == "combine_val":
        w = json.loads(a.weights)
        combine(vspecs, specs[0]["val_ids"], a.out, w)
    elif a.mode == "combine_test":
        w = json.loads(a.weights)
        combine(tspecs, specs[0]["test_ids"], a.out, w)


if __name__ == "__main__":
    main()
