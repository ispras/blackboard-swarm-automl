import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"


def valid(lo, la):
    return np.isfinite(lo) & np.isfinite(la) & (lo > -75.5) & (lo < -72.5) & (la > 39.5) & (la < 42.0)


def fix(lo, la):
    """Return corrected (lo,la) using candidate transforms if original invalid."""
    ok = valid(lo, la)
    cands = [(lo, la), (la, lo), (-lo, -la), (-la, -lo),
             (-lo, la), (lo, -la), (la, -lo), (-la, lo),
             (-np.abs(lo), np.abs(la)), (np.abs(lo), -np.abs(la)),
             (-np.abs(la), np.abs(lo)), (np.abs(la), -np.abs(lo))]
    out_lo = np.where(ok, lo, np.nan)
    out_la = np.where(ok, la, np.nan)
    for c_lo, c_la in cands:
        ck = valid(c_lo, c_la) & ~ok & np.isnan(out_lo)
        out_lo = np.where(ck, c_lo, out_lo)
        out_la = np.where(ck, c_la, out_la)
    return out_lo, out_la


def analyze(df, tag):
    plon = df["pickup_longitude"].to_numpy(np.float64)
    plat = df["pickup_latitude"].to_numpy(np.float64)
    dlon = df["dropoff_longitude"].to_numpy(np.float64)
    dlat = df["dropoff_latitude"].to_numpy(np.float64)
    pv = valid(plon, plat); dv = valid(dlon, dlat)
    bad = ~(pv & dv)
    fp_lo, fp_la = fix(plon, plat)
    fd_lo, fd_la = fix(dlon, dlat)
    rec = bad & valid(fp_lo, fp_la) & valid(fd_lo, fd_la)
    print(f"[{tag}] total={len(df)} bad={bad.sum()} recoverable={rec.sum()} still_bad={(bad & ~rec).sum()}")
    print(f"  fully_recovered_frac_of_bad={rec.sum()/max(bad.sum(),1):.4f}")
    return bad, rec


if __name__ == "__main__":
    t = pd.read_csv(f"{ROOT}/data/test.csv")
    b, r = analyze(t, "test")
    print(t[b & r][["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]].to_string())
