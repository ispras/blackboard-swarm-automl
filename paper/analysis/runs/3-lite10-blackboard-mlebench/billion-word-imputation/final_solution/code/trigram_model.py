import numpy as np
OUT='/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram'

class TrigramLM:
    def __init__(self, out=OUT):
        OUT=globals().get('OUT','/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram')
        OUT=out
        self.vocab=np.load(f'{OUT}/vocab.npy',allow_pickle=True)
        self.V=len(self.vocab); self.UNK=self.V; self.M=self.V+1
        self.vmap={w:i for i,w in enumerate(self.vocab)}
        self.U=np.load(f'{OUT}/U.npy'); self.N=float(self.U.sum())
        self.Kb=np.load(f'{OUT}/Kb.npy'); self.Cb=np.load(f'{OUT}/Cb.npy')
        self.Kt=np.load(f'{OUT}/Kt.npy'); self.Ct=np.load(f'{OUT}/Ct.npy').astype(np.float64)
        M=self.M
        # reverse bigram for predecessors
        id1=(self.Kb//M).astype(np.int64); id2=(self.Kb%M).astype(np.int64)
        Kr=id2*M+id1; o=np.argsort(Kr,kind='mergesort')
        self.Kr=Kr[o]; self.Cr=self.Cb[o]
        self.alpha=0.5
        self._succ={}; self._pred={}; self._tsucc={}
    def word(self,i): return self.vocab[i] if 0<=i<len(self.vocab) else '<unk>'
    def ids(self,toks):
        g=self.vmap.get
        return np.fromiter((g(t,self.UNK) for t in toks),dtype=np.int64,count=len(toks))
    def _top(self,keys,cnts,lo,hi,T):
        if hi<=lo: return np.empty(0,np.int64),np.empty(0,np.float64)
        ids=(keys[lo:hi]%self.M).astype(np.int64); cc=cnts[lo:hi]
        if len(ids)>T:
            sel=np.argpartition(cc,-T)[-T:]; ids=ids[sel]; cc=cc[sel]
        o=np.argsort(-cc); return ids[o],cc[o]
    def top_succ(self,a,T=40):
        r=self._succ.get((a,T))
        if r is None:
            r=self._top(self.Kb,self.Cb,a*self.M,(a+1)*self.M,T); self._succ[(a,T)]=r
        return r
    def top_pred(self,b,T=40):
        r=self._pred.get((b,T))
        if r is None:
            r=self._top(self.Kr,self.Cr,b*self.M,(b+1)*self.M,T); self._pred[(b,T)]=r
        return r
    def top_tsucc(self,x,a,T=30):
        r=self._tsucc.get((x,a,T))
        if r is None:
            base=(x*self.M+a)*self.M
            r=self._top(self.Kt,self.Ct,base,base+self.M,T); self._tsucc[(x,a,T)]=r
        return r
    def c2(self,a,b):
        a=np.asarray(a,dtype=np.int64); b=np.asarray(b,dtype=np.int64)
        key=a*self.M+b; idx=np.searchsorted(self.Kb,key); idx=np.clip(idx,0,len(self.Kb)-1)
        return np.where(self.Kb[idx]==key,self.Cb[idx].astype(np.float64),0.0)
    def c3(self,x,y,z):
        x=np.asarray(x,dtype=np.int64); y=np.asarray(y,dtype=np.int64); z=np.asarray(z,dtype=np.int64)
        key=(x*self.M+y)*self.M+z; idx=np.searchsorted(self.Kt,key); idx=np.clip(idx,0,len(self.Kt)-1)
        return np.where(self.Kt[idx]==key,self.Ct[idx],0.0)
    def score_gap(self,t, k, cand, al=None):
        # ids array t, gap before k (1<=k<=len-1)
        al=self.alpha if al is None else al
        a=int(t[k-1]); b=int(t[k])
        x=int(t[k-2]) if k>=2 else -1
        y=int(t[k+1]) if k+1<len(t) else -1
        cand=np.asarray(cand,dtype=np.int64)
        P1c=self.U[cand]/self.N; P1b=self.U[b]/self.N
        C2ac=self.c2(a,cand); C2ab=self.c2(np.array([a]),np.array([b]))[0]
        P2ca=(C2ac+al*P1c)/(self.U[a]+al*self.V)
        if x>=0:
            C3xac=self.c3(x,a,cand); C2xa=self.c2(np.array([x]),np.array([a]))[0]
            Pleft=(C3xac+al*P2ca)/(C2xa+al)
            P2ba=(C2ab+al*P1b)/(self.U[a]+al*self.V)
            C3xab=self.c3(np.array([x]),np.array([a]),np.array([b]))[0]
            base_left=(C3xab+al*P2ba)/(C2xa+al)
        else:
            Pleft=P2ca
            base_left=(C2ab+al*P1b)/(self.U[a]+al*self.V)
        C2cb=self.c2(cand,np.array([b])); C2cbv=np.broadcast_to(C2cb,cand.shape)
        P2bc=(C2cb+al*P1b)/(self.U[cand]+al*self.V)
        C3acb=self.c3(a,cand,np.array([b])); 
        Pmid=(C3acb+al*P2bc)/(C2ac+al)
        logscore=np.log(Pleft)+np.log(Pmid)
        logbase=np.log(base_left)
        if y>=0:
            P1y=self.U[y]/self.N; C2by=self.c2(np.array([b]),np.array([y]))[0]
            P2yb=(C2by+al*P1y)/(self.U[b]+al*self.V)
            C3cby=self.c3(cand,np.array([b]),np.array([y]))
            Pright=(C3cby+al*P2yb)/(C2cbv+al)
            C3aby=self.c3(np.array([a]),np.array([b]),np.array([y]))[0]
            base_y=(C3aby+al*P2yb)/(C2ab+al)
            logscore=logscore+np.log(Pright)
            logbase=logbase+np.log(base_y)
        gain=logscore-logbase
        return logscore,gain
    def predict_sentence(self,toks,Ts=40,Tp=40,Tt=30,topk=3):
        t=self.ids(toks); m=len(t)
        out=[]
        if m<3: return out
        for k in range(1,m):
            a=int(t[k-1]); b=int(t[k])
            si,_=self.top_succ(a,Ts); pi,_=self.top_pred(b,Tp)
            parts=[si,pi]
            if k>=2:
                ti,_=self.top_tsucc(int(t[k-2]),a,Tt); parts.append(ti)
            cand=np.unique(np.concatenate(parts))
            cand=cand[cand<self.V]  # drop unk
            if len(cand)==0: continue
            sc,gain=self.score_gap(t,k,cand)
            o=np.argsort(-gain)[:topk]
            for j in o:
                out.append((k,int(cand[j]),float(sc[j]),float(gain[j])))
        return out

if __name__=='__main__':
    lm=TrigramLM()
    print('V',lm.V,'bigrams',len(lm.Kb),'trigrams',len(lm.Kt))
    print(lm.predict_sentence('Ditto her cut-offs .'.split(' '))[:8])

def _refine_word(self, t, k, cap=1000):
    M=self.M
    a=int(t[k-1]); b=int(t[k])
    def topn(keys,cnts,base):
        lo=int(np.searchsorted(keys,base*M)); hi=int(np.searchsorted(keys,(base+1)*M))
        if hi<=lo: return np.empty(0,np.int64)
        ids=(keys[lo:hi]-base*M).astype(np.int64); cc=cnts[lo:hi]
        if len(ids)>cap:
            sel=np.argpartition(cc,-cap)[-cap:]; ids=ids[sel]
        return ids
    succ=topn(self.Kb,self.Cb,a)
    pred=topn(self.Kr,self.Cr,b)
    cand=np.unique(np.concatenate([succ,pred])); cand=cand[cand<self.V]
    if len(cand)==0: return None
    sc,gain=self.score_gap(t,k,cand)
    j=int(np.argmax(gain))
    return int(cand[j]), float(sc[j]), float(gain[j]), cand, sc, gain

TrigramLM.refine_word=_refine_word
