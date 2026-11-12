"""Finalizer: choose combiner (top-k prob-mean or softmax-weighted prob-mean)
by honest nested CV over all OOF members, then write final_solution/submission.csv.
"""
import os, glob, json
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

DATA='/download_job_code/payload/sandbox/data'
ROOT='/download_job_code/payload/sandbox'; WS=os.path.join(ROOT,'workspaces'); FINAL=os.path.join(ROOT,'final_solution')
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

def gauc(y,p): return roc_auc_score(y.ravel(), p.ravel())
def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))

def load_members():
    mem={}
    for f in glob.glob(os.path.join(WS,'*','**','oof*.npz'), recursive=True):
        try: z=np.load(f,allow_pickle=True)
        except Exception: continue
        if not ({"rec_ids","probs","test_rec_ids","test_probs"} <= set(z.files)): continue
        recs=list(z['rec_ids']); probs=np.asarray(z['probs'],np.float64)
        if probs.shape!=(len(train),19) or set(recs)!=set(train): continue
        P=probs[np.argsort([tr_i[r] for r in recs])]
        trecs=list(z['test_rec_ids']); tp=np.asarray(z['test_probs'],np.float64)
        if tp.shape!=(len(test),19) or set(trecs)!=set(test): continue
        T=tp[np.argsort([te_i[r] for r in trecs])]
        if not (np.isfinite(P).all() and np.isfinite(T).all()): continue
        mem[os.path.relpath(f,ROOT)]=(P,T)
    return mem

folds=[(tr,va) for s in range(10) for tr,va in KFold(5,shuffle=True,random_state=s).split(train)]

def member_auc(p): 
    try: return gauc(Y,p)
    except: return -1.0

def topk_weights(aucs, k):
    order=np.argsort(-np.asarray(aucs)); w=np.zeros(len(aucs))
    w[order[:k]]=1.0/k; return w
def softmax_weights(aucs, tau):
    a=np.asarray(aucs); w=np.exp((a-a.max())/tau); return w/w.sum()

def _agg(probs, w, space):
    V=np.tensordot(w, probs, axes=1)
    if space=='prob': return V
    eps=1e-6; P=np.clip(V,eps,1-eps)
    return np.log(P/(1-P))

def nested(keys, mem, combiner, param, space='prob'):
    scores=[]
    for tr,va in folds:
        ytr=Y[tr]
        aucs=[]
        for k in keys:
            try: aucs.append(gauc(ytr, mem[k][0][tr]))
            except Exception: aucs.append(-1.0)
        if combiner=='topk': w=topk_weights(aucs,param)
        else: w=softmax_weights(aucs,param)
        P=_agg(np.stack([mem[k][0][va] for k in keys]), w, space)
        scores.append(gauc(Y[va],P))
    return float(np.mean(scores)), float(np.std(scores))

def main():
    mem=load_members(); keys=list(mem)
    full_aucs=[member_auc(mem[k][0]) for k in keys]
    print("members",len(keys))
    best=None
    for space in ('prob','logit'):
      for k in [3,4,5,6,7,8,10,12,15,20,25,30,40,50,60,80,100]:
        if k>len(keys): continue
        ng,ns=nested(keys,mem,'topk',k,space)
        print(f"topk[{space}] k={k:3d} nested={ng:.4f}+/-{ns:.4f}",flush=True)
        if best is None or ng>best[0]: best=(ng,'topk',k,space,ns)
      for tau in [0.005,0.01,0.015,0.02,0.03,0.05,0.08]:
        ng,ns=nested(keys,mem,'softmax',tau,space)
        print(f"softmax[{space}] tau={tau} nested={ng:.4f}+/-{ns:.4f}",flush=True)
        if ng>best[0]: best=(ng,'softmax',tau,space,ns)
    print("BEST combiner:",best)
    ng_best,combiner,param,space,ns=best
    if combiner=='topk': w=topk_weights(full_aucs,param)
    else: w=softmax_weights(full_aucs,param)
    P=_agg(np.stack([mem[k][0] for k in keys]),w,space)
    m,g=mg(P)
    Traw=_agg(np.stack([mem[k][1] for k in keys]),w,space)
    if space=='logit':
        T=1/(1+np.exp(-Traw)); P=1/(1+np.exp(-P))
    else:
        T=Traw
    print(f"chosen {combiner}[{space}] param={param} full-OOF g={g:.4f} m={m:.4f} nested={ng_best:.4f}+/-{ns:.4f}")
    np.savez(os.path.join(FINAL,'blend_oof_test.npz'),rec_ids=np.array(train),probs=P,
             test_rec_ids=np.array(test),test_probs=T,weights=w,members=np.array(keys))
    with open(os.path.join(FINAL,'submission.csv'),'w') as f:
        f.write('Id,Probability\n')
        for r in test:
            for c in range(19): f.write(f'{r*100+c},{T[te_i[r],c]:.6f}\n')
    used=[keys[i] for i in np.argsort(-w) if w[i]>0][:80]
    json.dump({'combiner':combiner,'param':param,'space':space,'nested_auc':ng_best,'nested_std':ns,
               'full_oof_global':g,'full_oof_macro':m,'n_members':len(keys),'top_members':used},
              open(os.path.join(FINAL,'blend_report.json'),'w'),indent=2)
    print("wrote final_solution/submission.csv")

if __name__=='__main__': main()
