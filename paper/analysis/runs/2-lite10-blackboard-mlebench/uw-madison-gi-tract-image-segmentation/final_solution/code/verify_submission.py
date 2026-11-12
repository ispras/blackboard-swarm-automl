import sys, argparse, numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="final_solution/submission.csv"); ap.add_argument("--check_decode",type=int,default=1); args=ap.parse_args()
sub=pd.read_csv(args.csv, keep_default_na=False)
test=pd.read_csv("data/test.csv")
files=pd.read_parquet("workspaces/dubhe_zeta/test_files.parquet").set_index("id")
print("shape",sub.shape,"cols",list(sub.columns))
assert list(sub.columns)==["id","class","predicted"]
assert len(sub)==len(test), (len(sub),len(test))
# exact id,class multiset and order vs test.csv
assert (sub["id"].values==test["id"].values).all() and (sub["class"].values==test["class"].values).all(), "id/class order mismatch"
empty=(sub["predicted"].astype(str).str.strip()=="")
print("nonempty",int((~empty).sum()),"empty",int(empty.sum()),"frac",float((~empty).mean()))
bad=0; tot=0
for i,r in sub[~empty].iterrows():
    sid=r["id"]; h,w=int(files.loc[sid,"h"]),int(files.loc[sid,"w"]); N=h*w
    s=np.array(str(r["predicted"]).split(),dtype=np.int64)
    if len(s)==0 or len(s)%2!=0: bad+=1; continue
    st=s[0::2]; ln=s[1::2]
    if (st<1).any() or (ln<1).any(): bad+=1; continue
    if (np.diff(st)<=0).any(): bad+=1; continue
    end=st+ln-1
    if (end>N).any(): bad+=1; continue
    if len(st)>1 and (st[1:]<=end[:-1]).any(): bad+=1; continue
    tot+=1
print("RLE structural bad:",bad,"checked",tot)
if args.check_decode:
    sys.path.insert(0,"workspaces/dubhe_zeta")
    from common import rle_decode
    samp=sub[~empty].sample(min(200,(~empty).sum()),random_state=0)
    areas=[]
    for _,r in samp.iterrows():
        h,w=int(files.loc[r.id,"h"]),int(files.loc[r.id,"w"])
        m=rle_decode(r.predicted,(h,w)); assert m.shape==(h,w); areas.append(m.mean())
    print("decoded 200 ok; mean area frac",float(np.mean(areas)))
print("VERIFY OK")
