import sys, re, glob
import numpy as np, pandas as pd, cv2
from scipy import ndimage
ROOT="/download_job_code/payload/sandbox"
path=sys.argv[1] if len(sys.argv)>1 else f"{ROOT}/final_solution/submission.csv"
sub=pd.read_csv(path)
te=pd.read_csv(f"{ROOT}/data/test.csv")
print("rows",len(sub),"expected",len(te),"cols",list(sub.columns))
print("same (id,class) set", set(zip(sub.id,sub['class']))==set(zip(te.id,te['class'])))
print("order matches test.csv", len(sub)==len(te) and (sub.id.values==te.id.values).all() and (sub['class'].values==te['class'].values).all())
print("nonempty frac", round(sub.predicted.notna().mean(),4))
print("nonempty by class", sub.assign(nz=sub.predicted.notna()).groupby('class').nz.mean().round(4).to_dict())
# validity: sorted positive runs, no dup
bad=0
for r in sub[sub.predicted.notna()].itertuples(index=False):
    s=str(r.predicted).split()
    if len(s)%2: bad+=1; continue
    st=np.array(s[0::2],int); ln=np.array(s[1::2],int)
    if (ln<=0).any() or (st<=0).any() or (np.diff(st)<=0).any(): bad+=1
print("malformed RLE rows",bad)
# compactness on non-square sample under C, area distribution
cc=[]; areas={"large_bowel":[],"small_bowel":[],"stomach":[]}; n=0
for r in sub[sub.predicted.notna()].itertuples(index=False):
    m=re.match(r"(case\d+)_(day\d+)_slice_(\d+)$",r.id)
    if not m: continue
    g=glob.glob(f"{ROOT}/data/test/{m.group(1)}/{m.group(1)}_{m.group(2)}/scans/slice_{m.group(3)}_*.png")
    if not g: continue
    H,W=cv2.imread(g[0],cv2.IMREAD_UNCHANGED).shape
    s=str(r.predicted).split(); st=np.array(s[0::2],int)-1; ln=np.array(s[1::2],int)
    flat=np.zeros(H*W,np.uint8)
    for a,b in zip(st,ln): flat[a:a+b]=1
    mk=flat.reshape(H,W,order="C")
    _,k=ndimage.label(mk,np.ones((3,3))); 
    if H!=W: cc.append(k)
    areas[r._1 if hasattr(r,"_1") else r[1]].append(mk.sum()/(H*W))
    n+=1
    if n>=600: break
print("nonsquare mean comps (C)",round(float(np.mean(cc)),2) if cc else None,"sampled",n)
for k,v in areas.items(): print("area frac",k,round(float(np.median(v)),4), "p90",round(float(np.percentile(v,90)),4))
# compare with train GT positive fraction per class
tr=pd.read_csv(f"{ROOT}/data/train.csv"); print("train nonempty by class", tr.assign(nz=tr.segmentation.notna()).groupby('class').nz.mean().round(4).to_dict())
