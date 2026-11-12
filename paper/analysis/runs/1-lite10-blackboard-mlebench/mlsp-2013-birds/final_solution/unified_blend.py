"""altair_theta unified blend + finalizer.

Collects every oof*.npz with {rec_ids, probs, test_rec_ids, test_probs},
aligns to the canonical train/test order, and:
  - prints member OOF global/macro AUC
  - prob-mean of top-k for k sweep (full-OOF, optimistic)
  - honest nested top-k selection (3 seeds x 5-fold on train)
  - writes the chosen blend test predictions to final_solution
"""
import os, sys, glob, json
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

DATA='/download_job_code/payload/sandbox/data'
ROOT='/download_job_code/payload/sandbox'
WS=os.path.join(ROOT,'workspaces')
FINAL=os.path.join(ROOT,'final_solution')

labels={}
with open(f'{DATA}/essential_data/rec_labels_test_hidden.txt') as f:
    next(f)
    for line in f:
        p=line.strip().split(','); rid=int(p[0]); rest=[x for x in p[1:] if x!='']
        labels[rid]=('?' if (rest and rest[0]=='?') else set(int(x) for x in rest))
fold={}
with open(f'{DATA}/essential_data/CVfolds_2.txt') as f:
    next(f)
    for line in f:
        r,fo=line.strip().split(','); fold[int(r)]=int(fo)
train=sorted(r for r in fold if fold[r]==0); test=sorted(r for r in fold if fold[r]==1)
Y=np.zeros((len(train),19))
for i,r in enumerate(train):
    if isinstance(labels.get(r),set):
        for c in labels[r]: Y[i,c]=1
tr_i={r:i for i,r in enumerate(train)}; te_i={r:i for i,r in enumerate(test)}
Yte=None

def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))

def load_members():
    mem={}
    for f in glob.glob(os.path.join(WS,'*','**','oof*.npz'), recursive=True):
        try: z=np.load(f,allow_pickle=True)
        except Exception: continue
        if not ({"rec_ids","probs","test_rec_ids","test_probs"} <= set(z.files)): continue
        recs=list(z['rec_ids']); probs=np.asarray(z['probs'],dtype=np.float64)
        if probs.shape!=(len(train),19): continue
        if set(recs)!=set(train): continue
        order=np.argsort([tr_i[r] for r in recs])
        P=probs[order]
        trecs=list(z['test_rec_ids']); tp=np.asarray(z['test_probs'],dtype=np.float64)
        if tp.shape!=(len(test),19) or set(trecs)!=set(test): continue
        o=np.argsort([te_i[r] for r in trecs]); T=tp[o]
        key=os.path.relpath(f, ROOT)
        mem[key]=(P,T)
    return mem

def nested_select(mem, keys, k):
    scores=[]
    for seed in (0,1,2):
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
            # rank members on tr OOF
            sc=[]
            for kk in keys:
                p=mem[kk][0][tr]
                try: a=roc_auc_score(Y[tr].ravel(), p.ravel())
                except Exception: a=-1
                sc.append((a,kk))
            sc.sort(reverse=True)
            sel=[kk for _,kk in sc[:k]]
            pavg=np.mean([mem[kk][0][va] for kk in sel],0)
            scores.append(roc_auc_score(Y[va].ravel(), pavg.ravel()))
    return float(np.mean(scores)), float(np.std(scores))

def main():
    mem=load_members()
    print("members:",len(mem))
    scored=[]
    for k,(P,T) in mem.items():
        m,g=mg(P); scored.append((g,m,k))
    scored.sort(reverse=True)
    for g,m,k in scored:
        print(f"  g={g:.4f} m={m:.4f}  {k}")
    keys=[k for _,_,k in scored]
    print("\n-- prob-mean top-k (full-OOF) --")
    for k in [1,3,5,8,10,12,15,20,25,30,40,50,len(keys)]:
        if k>len(keys): continue
        P=np.mean([mem[kk][0] for kk in keys[:k]],0)
        m,g=mg(P); print(f"  k={k:3d} g={g:.4f} m={m:.4f}")
    print("\n-- honest nested top-k --")
    best=(None,-1)
    for k in [5,8,10,12,15,20,25,30,40,50]:
        if k>len(keys): continue
        ng,ns=nested_select(mem,keys,k)
        print(f"  k={k:3d} nested g={ng:.4f} +/- {ns:.4f}")
        if ng>best[1]: best=(k,ng)
    k=best[0]
    print("chosen k =",k)
    P=np.mean([mem[kk][0] for kk in keys[:k]],0)
    T=np.mean([mem[kk][1] for kk in keys[:k]],0)
    m,g=mg(P)
    print(f"chosen blend full-OOF g={g:.4f} m={m:.4f}, nested={best[1]:.4f}")
    np.savez(os.path.join(FINAL,'blend_oof_test.npz'), rec_ids=np.array(train), probs=P,
             test_rec_ids=np.array(test), test_probs=T, chosen_k=k, members=np.array(keys[:k]))
    # write submission
    with open(os.path.join(FINAL,'submission.csv'),'w') as f:
        f.write('Id,Probability\n')
        for r in test:
            for c in range(19):
                f.write(f'{r*100+c},{T[te_i[r],c]:.6f}\n')
    print("wrote final_solution/submission.csv from top", k)
    json.dump({'chosen_k':k,'full_oof_global':g,'full_oof_macro':m,'nested':best[1],
               'members':keys[:k]}, open(os.path.join(FINAL,'blend_report.json'),'w'), indent=2)

if __name__=='__main__':
    main()
