import numpy as np, pandas as pd
ROOT="/download_job_code/payload/sandbox"
idx=pd.read_csv(f"{ROOT}/workspaces/aldebaran_omicron/probs/test_A_AC_probs_index.csv")
ids=list(idx.id); idpos={x:i for i,x in enumerate(ids)}
families={
 "A":[("workspaces/aldebaran_omicron/probs/test_A_AC_probs.npy","workspaces/aldebaran_omicron/probs/test_A_AC_probs_index.csv"),
      ("workspaces/aldebaran_omicron/probs/test_A_AC_alldata_probs.npy","workspaces/aldebaran_omicron/probs/test_A_AC_alldata_probs_index.csv")],
 "B":[("workspaces/regulus_alpha/probs/test_B2_ep6_probs.npy","workspaces/regulus_alpha/probs/test_B2_ep6_ids.csv"),
      ("workspaces/regulus_alpha/probs/test_B2_alldata320_probs.npy","workspaces/regulus_alpha/probs/test_B2_alldata320_ids.csv")],
 "C":[("workspaces/canopus_phi/test_C3_probs.npy","workspaces/canopus_phi/test_C3_probs_ids.csv"),
      ("workspaces/canopus_phi/test_C3alldata_probs.npy","workspaces/canopus_phi/test_C3alldata_probs_ids.csv")],
}
acc=np.zeros((len(ids),3,320,320),np.float32); nfam=0
for fam,files in families.items():
    facc=np.zeros_like(acc); cnt=0
    for p,idf in files:
        pr=np.load(f"{ROOT}/{p}",mmap_mode="r"); idfdf=pd.read_csv(f"{ROOT}/{idf}")
        col="id" if "id" in idfdf.columns else idfdf.columns[0]
        mp=dict(zip(idfdf[col],range(len(idfdf))))
        hit=0
        for rid,i in idpos.items():
            j=mp.get(rid)
            if j is not None: facc[i]+=pr[j].astype(np.float32); hit+=1
        print(fam,p,pr.shape,"hit",hit)
        cnt+=1
    facc/=cnt; acc+=facc; nfam+=1
    print("family done",fam)
acc/=nfam
mm=np.lib.format.open_memmap(f"{ROOT}/workspaces/aldebaran_omicron/probs/test_final_candidate_raw.npy",mode="w+",dtype=np.uint8,shape=acc.shape)
mm[:]=acc.round().clip(0,255).astype(np.uint8); mm.flush(); print("wrote candidate raw")
