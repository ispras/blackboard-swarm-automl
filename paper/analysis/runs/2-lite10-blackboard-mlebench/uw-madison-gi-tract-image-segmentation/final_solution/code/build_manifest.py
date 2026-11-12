import pandas as pd, numpy as np, glob, os, re
from PIL import Image

def build(split):
    rows=[]
    base=f"data/{split}"
    for f in glob.glob(base+"/*/*/scans/*.png"):
        fn=os.path.basename(f)
        m=re.match(r"slice_(\d+)_(\d+)_(\d+)_([\d.]+)_([\d.]+)\.png",fn)
        idx=int(m.group(1)); W=int(m.group(2)); H=int(m.group(3))
        sw=float(m.group(4)); sh=float(m.group(5))
        case_day=os.path.basename(os.path.dirname(os.path.dirname(f)))
        case=case_day.split("_day")[0]
        rows.append(dict(id=f"{case_day}_slice_{idx:04d}", path=f, case_day=case_day, case=case,
                         idx=idx, h=H, w=W, sw=sw, sh=sh))
    return pd.DataFrame(rows)

trf=build("train"); tef=build("test")
# sanity: verify parsed h,w vs PIL for a sample
for df in (trf,tef):
    for _,r in df.sample(20,random_state=0).iterrows():
        W,H=Image.open(r.path).size
        assert (H,W)==(int(r.h),int(r.w)), (r.path,H,W,r.h,r.w)
print("train files",len(trf),"test files",len(tef),"h/w verified")
tr=pd.read_csv("data/train.csv").merge(trf[["id","path","case_day","case","idx","h","w"]],on="id",how="left")
assert tr.path.notna().all()
tr.to_parquet("workspaces/dubhe_zeta/train_rows.parquet")
trf.to_parquet("workspaces/dubhe_zeta/train_files.parquet")
te=pd.read_csv("data/test.csv").merge(tef[["id","path","case_day","case","idx","h","w"]],on="id",how="left")
assert te.path.notna().all()
te.to_parquet("workspaces/dubhe_zeta/test_rows.parquet")
tef.to_parquet("workspaces/dubhe_zeta/test_files.parquet")
print(tr.groupby(["h","w"]).size().to_dict())
