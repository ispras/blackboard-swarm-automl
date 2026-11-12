import numpy as np, pandas as pd
mine_idx=pd.read_csv("workspaces/aldebaran_omicron/probs/test_A_AC_probs_index.csv")
models=[("workspaces/aldebaran_omicron/probs/test_A_AC_probs.npy","workspaces/aldebaran_omicron/probs/test_A_AC_probs_index.csv"),
        ("workspaces/regulus_alpha/probs/test_B2_ep6_probs.npy","workspaces/regulus_alpha/probs/test_B2_ep6_ids.csv"),
        ("workspaces/regulus_alpha/probs/test_C3.npy","workspaces/regulus_alpha/probs/test_C3_ids.csv")]
out=np.zeros((len(mine_idx),3,320,320),np.float32); cnt=np.zeros(len(mine_idx))
for p,idf in models:
    pr=np.load(p,mmap_mode="r"); ids=pd.read_csv(idf); idcol="id" if "id" in ids.columns else ids.columns[0]
    mp={x:i for i,x in enumerate(ids[idcol])}
    for k,rid in enumerate(mine_idx.id):
        if rid in mp: out[k]+=pr[mp[rid]].astype(np.float32); cnt[k]+=1
    print(p,pr.shape,"matched",int((cnt>0).sum()))
out/=np.maximum(cnt,1)[:,None,None,None]
print("coverage min/max",cnt.min(),cnt.max())
mm=np.lib.format.open_memmap("workspaces/aldebaran_omicron/probs/test_fused3_check.npy",mode="w+",dtype=np.uint8,shape=out.shape)
mm[:]=out.round().clip(0,255).astype(np.uint8); mm.flush(); print("wrote test fused3")
