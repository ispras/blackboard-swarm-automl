import numpy as np, pandas as pd, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_lib import load_helmets, build_index, nearest_box

VIEWS = ["Sideline", "Endzone"]
TH = 2.5
OUT = "workspaces/alkaid_psi"


def pair_positions(df, tracking_cols):
    """df needs game_play, step, p1, p2. adds x1,y1,x2,y2,dist."""
    tr = pd.read_csv("data/train_player_tracking.csv", usecols=tracking_cols)
    return _merge(df, tr)


def _merge(df, tr):
    a = tr.rename(columns={"nfl_player_id": "p1", "x_position": "x1", "y_position": "y1"})
    df = df.merge(a, on=["game_play", "step", "p1"], how="left")
    b = tr.rename(columns={"nfl_player_id": "p2", "x_position": "x2", "y_position": "y2"})
    df = df.merge(b, on=["game_play", "step", "p2"], how="left")
    df["dist"] = np.sqrt((df.x1 - df.x2) ** 2 + (df.y1 - df.y2) ** 2)
    return df


def add_boxes(df, hel):
    idx = build_index(hel)
    n = len(df)
    for v in VIEWS:
        for k in range(4):
            df[f"{v}_b1_{k}"] = np.float32(0)
            df[f"{v}_b2_{k}"] = np.float32(0)
        df[f"{v}_nv"] = np.int8(0)
    nv1 = np.zeros((n, 2), np.int8)
    for view_i, v in enumerate(VIEWS):
        c1 = np.zeros((n, 4), np.float32)
        c2 = np.zeros((n, 4), np.float32)
        for i, (gp, fr, p1, p2) in enumerate(zip(df.game_play.values, df.frame.values, df.p1.values, df.p2.values)):
            b1 = nearest_box(idx, gp, v, p1, fr)
            b2 = nearest_box(idx, gp, v, p2, fr)
            if b1 is not None:
                c1[i] = b1
            if b2 is not None:
                c2[i] = b2
            nv1[i, view_i] = int(b1 is not None) + int(b2 is not None)
        for k in range(4):
            df[f"{v}_b1_{k}"] = c1[:, k]
            df[f"{v}_b2_{k}"] = c2[:, k]
        df[f"{v}_nv"] = nv1[:, view_i]
    return df


def build_train(hel):
    lab = pd.read_csv("data/train_labels.csv",
                      usecols=["contact_id","game_play","step","nfl_player_id_1","nfl_player_id_2","contact"])
    pp = lab[lab.nfl_player_id_2 != "G"].copy()
    pp["p1"] = pp.nfl_player_id_1.astype("int32")
    pp["p2"] = pd.to_numeric(pp.nfl_player_id_2).astype("int32")
    pp = _merge(pp, pd.read_csv("data/train_player_tracking.csv",
                                usecols=["game_play","step","nfl_player_id","x_position","y_position"]))
    pp["frame"] = (300 + 5.994 * pp.step).round().astype(int)
    cand = pp[pp.dist <= TH].copy()
    fold = pd.read_csv(f"{OUT}/cv_folds.csv")
    cand = cand.merge(fold, on="game_play", how="left")
    cand = cand[["contact_id","game_play","step","frame","p1","p2","dist","contact","fold"]].reset_index(drop=True)
    cand = add_boxes(cand, hel)
    cand.to_parquet(f"{OUT}/train_cand.parquet", index=False)
    # keep all pp rows info to expand predictions later
    pp[["contact_id","game_play","step","p1","p2","dist","contact"]].merge(fold, on="game_play", how="left") \
        .to_parquet(f"{OUT}/train_pp_rows.parquet", index=False)
    print("train cand", len(cand), "pos", int(cand.contact.sum()), "plays", cand.game_play.nunique())


def build_test(hel):
    sub = pd.read_csv("data/sample_submission.csv")
    parts = sub.contact_id.str.rsplit("_", n=3, expand=True)
    sub["game_play"] = parts[0]
    sub["step"] = parts[1].astype("int32")
    sub["p1"] = parts[2].astype("int32")
    sub["p2s"] = parts[3]
    pp = sub[sub.p2s != "G"].copy()
    pp["p2"] = pp.p2s.astype("int32")
    pp = _merge(pp, pd.read_csv("data/test_player_tracking.csv",
                                usecols=["game_play","step","nfl_player_id","x_position","y_position"]))
    pp["frame"] = (300 + 5.994 * pp.step).round().astype(int)
    cand = pp[pp.dist <= TH].copy()
    cand = cand[["contact_id","game_play","step","frame","p1","p2","dist"]].reset_index(drop=True)
    cand = add_boxes(cand, hel)
    cand.to_parquet(f"{OUT}/test_cand.parquet", index=False)
    pp[["contact_id","game_play","step","p1","p2","dist"]].to_parquet(f"{OUT}/test_pp_rows.parquet", index=False)
    print("test pp", len(pp), "cand", len(cand), "plays", cand.game_play.nunique(),
          "cand frac", len(cand) / len(pp))


if __name__ == "__main__":
    t = time.time()
    hel = load_helmets(VIEWS)
    print("helmets", len(hel), time.time() - t)
    build_train(hel)
    hel_test = load_helmets(VIEWS, "data/test_baseline_helmets.csv")
    print("test helmets", len(hel_test))
    build_test(hel_test)
    print("done", time.time() - t)
