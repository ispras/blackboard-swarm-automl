import json
import numpy as np
import pandas as pd

T = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
BLEND = "workspaces/mirfak_zeta/blend_test.npy"


def main():
    blend = np.load(BLEND).astype(np.float64)  # (240,107,5)
    # unscored cols (indices 2,4) not modeled -> use scored analogues as proxies
    blend[..., 2] = blend[..., 1]
    blend[..., 4] = blend[..., 3]

    test = [json.loads(l) for l in open("data/test.json") if l.strip()]
    samp = pd.read_csv("data/sample_submission.csv")
    ids = samp["id_seqpos"].tolist()
    id_to_idx = {x["id"]: i for i, x in enumerate(test)}
    rows = np.zeros((len(ids), 5), dtype=np.float64)
    for k, idseq in enumerate(ids):
        sid, pos = idseq.rsplit("_", 1)
        rows[k] = blend[id_to_idx[sid], int(pos)]
    out = samp.copy()
    for j, t in enumerate(T):
        out[t] = rows[:, j]
    out.to_csv("final_solution/submission.csv", index=False)
    print("wrote final_solution/submission.csv", out.shape)


if __name__ == "__main__":
    main()
