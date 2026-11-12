import sys, os, time, numpy as np
TRAIN='/download_job_code/payload/sandbox/workspaces/antares_upsilon/train_v2.txt'
OUT=sys.argv[2] if len(sys.argv)>2 else '/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram'
N_LINES=int(sys.argv[1]) if len(sys.argv)>1 else 30_000_000
vocab=np.load(f'{OUT}/vocab.npy',allow_pickle=True)
V=len(vocab); UNK=V; M=V+1
vmap={w:i for i,w in enumerate(vocab)}
t0=time.time()

def reduce_pairs(keys_list):
    K=np.concatenate(keys_list)
    o=np.argsort(K,kind='mergesort'); K=K[o]
    newgrp=np.empty(len(K),dtype=bool); newgrp[0]=True; newgrp[1:]=K[1:]!=K[:-1]
    idx=np.flatnonzero(newgrp)
    C=np.add.reduceat(np.ones(len(K),dtype=np.int64),idx) if False else None
    # counts: since each chunk already unique, duplicates across chunks -> recompute via bincount-like
    # use np.unique on keys to sum? We stored counts separately. Rebuild with counts:
    return K, idx

# We'll carry counts too
def reduce_kc(keys_list,cnts_list):
    K=np.concatenate(keys_list); C=np.concatenate(cnts_list)
    o=np.argsort(K,kind='mergesort'); K=K[o]; C=C[o]
    newgrp=np.empty(len(K),dtype=bool); newgrp[0]=True; newgrp[1:]=K[1:]!=K[:-1]
    idx=np.flatnonzero(newgrp)
    C2=np.add.reduceat(C,idx)
    return K[idx],C2

SUPER=500_000  # lines per superchunk
finalK=[]; finalC=[]
superK=[]; superC=[]
buf=[]
def flush_buf():
    if not buf: return
    keys=np.concatenate(buf)
    uk,ck=np.unique(keys,return_counts=True)
    superK.append(uk.astype(np.int64)); superC.append(ck.astype(np.int64)); buf.clear()

with open(TRAIN,encoding='utf-8') as f:
    for i,line in enumerate(f):
        if i>=N_LINES: break
        toks=line.rstrip('\n').split(' ')
        n=len(toks)
        if n>=3:
            ids=np.fromiter((vmap.get(t,UNK) for t in toks),dtype=np.int64,count=n)
            buf.append(ids[:-2]*M*M + ids[1:-1]*M + ids[2:])
        if (i+1)%100_000==0:
            flush_buf()
        if (i+1)%SUPER==0:
            K,C=reduce_kc(superK,superC); finalK.append(K); finalC.append(C)
            superK=[]; superC=[]; buf=[]
            print('  merged at',i+1,'final unique approx',sum(len(x) for x in finalK),round(time.time()-t0,1),flush=True)
    flush_buf()
    if superK:
        K,C=reduce_kc(superK,superC); finalK.append(K); finalC.append(C)
Kt,Ct=reduce_kc(finalK,finalC)
np.save(f'{OUT}/Kt.npy',Kt); np.save(f'{OUT}/Ct.npy',Ct.astype(np.int32))
print('trigram done',len(Kt),'unique',time.time()-t0)
