import sys, os, time, numpy as np
from collections import Counter

TRAIN='/download_job_code/payload/sandbox/workspaces/antares_upsilon/train_v2.txt'
OUT=sys.argv[3] if len(sys.argv)>3 else '/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram'
N_LINES=int(sys.argv[1]) if len(sys.argv)>1 else 6_000_000
V=int(sys.argv[2]) if len(sys.argv)>2 else 120000
os.makedirs(OUT, exist_ok=True)
t0=time.time()

# pass 1: unigram
uc=Counter()
with open(TRAIN,encoding='utf-8') as f:
    for i,line in enumerate(f):
        if i>=N_LINES: break
        uc.update(line.rstrip('\n').split(' '))
print('pass1 done',time.time()-t0,'unique',len(uc))
vocab=[w for w,_ in uc.most_common(V)]
vmap={w:i for i,w in enumerate(vocab)}
UNK=V
U=np.zeros(V+1,dtype=np.int64)
for w,c in uc.items():
    U[vmap.get(w,UNK)]+=c
del uc
np.save(f'{OUT}/vocab.npy',np.array(vocab,dtype=object),allow_pickle=True)
np.save(f'{OUT}/U.npy',U)
print('vocab saved',V,'unk frac',U[UNK]/U.sum(),time.time()-t0)

# pass 2: bigram
CHUNK=200_000
allkeys=[]; allcnts=[]
buf_keys=[]
def flush(buf):
    if not buf: return
    keys=np.concatenate(buf)
    uk,ck=np.unique(keys,return_counts=True)
    allkeys.append(uk.astype(np.int64)); allcnts.append(ck.astype(np.int64))
    buf.clear()
with open(TRAIN,encoding='utf-8') as f:
    buf=[]
    for i,line in enumerate(f):
        if i>=N_LINES: break
        toks=line.rstrip('\n').split(' ')
        ids=np.fromiter((vmap.get(t,UNK) for t in toks),dtype=np.int32,count=len(toks))
        if len(ids)>=2:
            buf.append(ids[:-1].astype(np.int64)*(V+1)+ids[1:])
        if (i+1)%CHUNK==0:
            flush(buf)
            if ((i+1)//CHUNK)%10==0: print('  pass2',i+1,time.time()-t0)
    flush(buf)
Kb=np.concatenate(allkeys); Cb=np.concatenate(allcnts)
del allkeys,allcnts
order=np.argsort(Kb,kind='mergesort'); Kb=Kb[order]; Cb=Cb[order]
# reduce duplicates across chunks
if len(Kb)>0:
    newgrp=np.empty(len(Kb),dtype=bool); newgrp[0]=True; newgrp[1:]=Kb[1:]!=Kb[:-1]
    idx=np.flatnonzero(newgrp)
    Cb=np.add.reduceat(Cb,idx)
    Kb=Kb[idx]
np.save(f'{OUT}/Kb.npy',Kb); np.save(f'{OUT}/Cb.npy',Cb)
print('bigram done',len(Kb),'unique',time.time()-t0)
