import sys
import numpy as np, pandas as pd


def rle_decode(rle, H, W):
    flat = np.zeros(H * W, np.uint8)
    if isinstance(rle, str) and len(rle):
        s = rle.split()
        st = np.asarray(s[0::2], np.int64) - 1
        ln = np.asarray(s[1::2], np.int64)
        assert (ln > 0).all(), "non-positive run length"
        for a, b in zip(st, ln):
            assert a >= 0 and a + b <= H * W, "run out of bounds"
            flat[a:a + b] = 1
    return flat.reshape((H, W), order="C")


def validate(sub_csv, sample_csv):
    sub = pd.read_csv(sub_csv, keep_default_na=False)
    ss = pd.read_csv(sample_csv, keep_default_na=False)
    assert list(sub.columns) == ["id", "class", "predicted"], sub.columns
    assert len(sub) == len(ss), (len(sub), len(ss))
    assert (sub["id"].values == ss["id"].values).all(), "id order mismatch"
    assert (sub["class"].values == ss["class"].values).all(), "class order mismatch"
    for i, r in sub.iterrows():
        s = str(r["predicted"]).strip()
        if s == "":
            continue
        v = s.split()
        assert len(v) % 2 == 0, f"odd tokens row {i}"
        st = list(map(int, v[0::2])); ln = list(map(int, v[1::2]))
        assert all(x > 0 for x in st), f"non-positive start row {i}"
        assert all(x > 0 for x in ln), f"non-positive length row {i}"
        ends = [a + b - 1 for a, b in zip(st, ln)]
        assert all(ends[k] < st[k + 1] for k in range(len(st) - 1)), f"overlap/unsorted row {i}"
    print("OK: rows", len(sub), "non-empty", (sub['predicted'] != '').sum(),
          "classes", sub['class'].value_counts().to_dict())


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
             "/download_job_code/payload/sandbox/data/sample_submission.csv")
