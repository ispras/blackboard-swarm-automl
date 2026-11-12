"""Create submission.csv from a (7545,80) test prediction array (breath-major, id-sorted)."""
import sys
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"


def make(pred_path, out_path=f"{ROOT}/final_solution/submission.csv", clip=True):
    p = np.load(pred_path)
    te = pd.read_csv(f"{ROOT}/data/test.csv", usecols=["id", "breath_id"]).sort_values(["breath_id", "id"]).reset_index(drop=True)
    assert p.shape == (7545, 80), p.shape
    assert len(te) == 7545 * 80
    flat = p.reshape(-1)
    if clip:
        flat = np.clip(flat, -2.0, 65.0)
    sub = pd.DataFrame({"id": te["id"].values, "pressure": flat})
    sub = sub.sort_values("id")
    sample = pd.read_csv(f"{ROOT}/data/sample_submission.csv", usecols=["id"])
    sub = sample.merge(sub, on="id", how="left")
    assert sub.pressure.notna().all()
    sub.to_csv(out_path, index=False)
    print("wrote", out_path, sub.shape, "mean", sub.pressure.mean(), "min", sub.pressure.min(), "max", sub.pressure.max())


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/workspaces/canopus_iota/test_pred_lstm384.npy")
