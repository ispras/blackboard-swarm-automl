import numpy as np, common as C
tr=C.load_train(); te=C.load_test(); Y=C.make_targets(tr)
BI={b:i for i,b in enumerate("ACGU")}
def oh(seqs,L):
    A=np.zeros((len(seqs),L),np.uint8)
    for i,s in enumerate(seqs):
        for j in range(min(L,len(s))): A[i,j]=BI.get(s[j],0)
    return A
def ident(A,B): return (A[:,None,:]==B[None,:,:]).mean(-1).astype(np.float32)
A68=oh([r["sequence"][:68] for r in tr],68); B68=oh([r["sequence"][:68] for r in te],68)
A107=oh([r["sequence"] for r in tr],107); B107=oh([r["sequence"] for r in te],107)
S_tr=(0.7*ident(A68,A68)+0.3*ident(A107,A107)).astype(np.float32)
S_te=(0.7*ident(B68,A68)+0.3*ident(B107,A107)).astype(np.float32)
# for train exclude self-ish (self mix=1.0 exactly; subtract tiny)
np.fill_diagonal(S_tr,-1.0)
def knn(S,k=3,p=5):
    N=S.shape[0]; idx=np.argpartition(-S,kth=k,axis=1)[:,:k]
    out=np.zeros((N,68,5),np.float32)
    for i in range(N):
        nb=idx[i]; s=S[i,nb]; w=np.where(s>0,s**p,0)
        if w.sum()>0: out[i]=np.tensordot(w,Y[nb],axes=(0,0))/w.sum()
    return out
oof=knn(S_tr); test=np.zeros((240,107,5),np.float32); test[:,:68,:]=knn(S_te)
np.save("knn_best_oof.npy",oof); np.save("knn_best_test.npy",test)
np.save("knn_best_S_tr.npy",S_tr); np.save("knn_best_S_te.npy",S_te)
sn=np.array([r["SN_filter"] for r in tr],dtype=bool)
print("knn alone sn1", C.mcrmse(Y[sn],oof[sn])[0])
print("test id mix >=0.8:", int((S_te.max(1)>=0.8).sum()), ">=0.9:", int((S_te.max(1)>=0.9).sum()))
