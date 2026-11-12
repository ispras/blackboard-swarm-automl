import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_lib import load_helmets, build_index, nearest_box

VIEWS = ["Sideline", "Endzone"]
OUT = "workspaces/alkaid_psi"
IMH = 720.0


def add_boxes(df, hel):
    idx = build_index(hel)
    n = len(df)
    for v in VIEWS:
        for k in range(4):
            df[f"{v}_b_{k}"] = np.float32(0)
        df[f"{v}_nv"] = np.int8(0)
    for vi, v in enumerate(VIEWS):
        c = np.zeros((n, 4), np.float32)
        nv = np.zeros(n, np.int8)
        for i, (gp, fr, pid) in enumerate(zip(df.game_play.values, df.frame.values, df.player.values)):
            b = nearest_box(idx, gp, v, pid, fr)
            if b is not None:
                c[i] = b
                nv[i] = 1
        for k in range(4):
            df[f"{v}_b_{k}"] = c[:, k]
        df[f"{v}_nv"] = nv
    return df


def build_train(hel):
    lab = pd.read_csv("data/train_labels.csv",
                      usecols=["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"])
    g = lab[lab.nfl_player_id_2.astype(str) == "G"].copy()
    g["player"] = g.nfl_player_id_1.astype("int32")
    g["frame"] = (300 + 5.994 * g.step).round().astype(int)
    fold = pd.read_csv(f"{OUT}/cv_folds.csv")
    g = g.merge(fold, on="game_play", how="left")
    g = g[["contact_id", "game_play", "step", "frame", "player", "contact", "fold"]].reset_index(drop=True)
    g = add_boxes(g, hel)
    g.to_parquet(f"{OUT}/train_ground.parquet", index=False)
    print("train ground", len(g), "pos", int(g.contact.sum()), "plays", g.game_play.nunique())


def build_test(hel):
    sub = pd.read_csv("data/sample_submission.csv")
    parts = sub.contact_id.str.rsplit("_", n=3, expand=True)
    out = pd.DataFrame({"contact_id": sub.contact_id, "game_play": parts[0], "step": parts[1].astype("int32"),
                        "player": parts[2].astype("int32"), "p2s": parts[3]})
    g = out[out.p2s == "G"].copy()
    g["frame"] = (300 + 5.994 * g.step).round().astype(int)
    g = g[["contact_id", "game_play", "step", "frame", "player"]].reset_index(drop=True)
    g = add_boxes(g, hel)
    g.to_parquet(f"{OUT}/test_ground.parquet", index=False)
    print("test ground", len(g), "plays", g.game_play.nunique())


if __name__ == "__main__":
    build_train(load_helmets(VIEWS))
    build_test(load_helmets(VIEWS, "data/test_baseline_helmets.csv"))
