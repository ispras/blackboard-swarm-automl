import numpy as np, cv2, pandas as pd


def load_helmets(views=("Sideline", "Endzone"), path="data/train_baseline_helmets.csv"):
    h = pd.read_csv(path,
                    usecols=["game_play", "view", "frame", "nfl_player_id",
                             "left", "width", "top", "height"])
    h = h[h.view.isin(views)]
    h["cx"] = h.left + h.width / 2.0
    h["cy"] = h.top + h.height / 2.0
    h["nfl_player_id"] = h.nfl_player_id.astype("int32")
    return h


def build_index(hel):
    """dict[(game_play,view,player)] -> (frames int array, boxes float array Nx4(cx,cy,w,h))"""
    idx = {}
    for (gp, view, pid), g in hel.groupby(["game_play", "view", "nfl_player_id"], sort=False):
        g = g.sort_values("frame")
        idx[(gp, view, int(pid))] = (
            g.frame.values.astype(np.int32),
            np.stack([g.cx.values, g.cy.values, g.width.values, g.height.values], 1).astype(np.float32),
        )
    return idx


def nearest_box(idx, gp, view, pid, frame, tol=6):
    key = (gp, view, int(pid))
    if key not in idx:
        return None
    fr, bx = idx[key]
    i = np.searchsorted(fr, frame)
    best = None
    bestd = tol + 1
    for j in (i - 1, i):
        if 0 <= j < len(fr):
            d = abs(int(fr[j]) - frame)
            if d < bestd:
                bestd = d
                best = bx[j]
    return best


def params_from_boxes(b1, b2, variant="T"):
    """return (cx,cy,side, nvalid) for a square crop covering the pair."""
    if b1 is None and b2 is None:
        return 0.0, 0.0, 0.0, 0
    if b1 is not None and b2 is not None:
        cx = (b1[0] + b2[0]) / 2.0
        cy = (b1[1] + b2[1]) / 2.0
        d = ((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2) ** 0.5
        m = max(b1[2], b1[3], b2[2], b2[3])
        if variant == "W":
            side = max(2.6 * d + 1.0 * (b1[2] + b1[3] + b2[2] + b2[3]) / 2.0, 3.8 * m, 72.0)
        else:
            side = max(1.45 * d + 0.55 * (b1[2] + b1[3] + b2[2] + b2[3]) / 2.0, 2.4 * m, 40.0)
        return cx, cy, side, 2
    b = b1 if b1 is not None else b2
    return b[0], b[1], max(2.4 * max(b[2], b[3]), 40.0), 1


def crop(frame, cx, cy, side, S):
    side = int(max(16, min(round(side), 900)))
    patch = cv2.getRectSubPix(frame, (side, side), (float(cx), float(cy)))
    if patch.shape[0] != S or patch.shape[1] != S:
        patch = cv2.resize(patch, (S, S), interpolation=cv2.INTER_AREA)
    return patch


def geometry_features(b1, b2):
    """numeric per-view features from the two boxes (any may be None)."""
    f = np.zeros(12, dtype=np.float32)
    if b1 is not None:
        f[0:4] = b1
        f[4] = 1
    if b2 is not None:
        f[5:9] = b2
        f[9] = 1
    if b1 is not None and b2 is not None:
        d = ((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2) ** 0.5
        m = max(b1[2], b1[3], b2[2], b2[3])
        f[10] = d
        f[11] = d / (m + 1e-3)
    return f, int(b1 is not None) + int(b2 is not None)
