"""Apply fused gap ranker + gated expected-Levenshtein medoid to the 302,644 test ids.
Streams MLM shards + fomalhaut top-10 gap posteriors + fomalhaut top-3 trigram words."""
import json, csv, io, math, pickle, sys, zipfile, glob
import numpy as np

cfg=json.load(open('fusion_config.json'))
clf=pickle.load(open('fusion_model.pkl','rb'))
TH=cfg['threshold']; ALPHA=cfg['alpha']; MPB=cfg['mpb']; TOPN=cfg['topN']
PUNCT=set([',','.',';',':','!','?','"',"'",'(',')','--','-','``',"''"])

SHARDS=sorted(glob.glob('mlm_test_*.jsonl'), key=lambda p:int(p.split('_')[2]))
FOM='../fomalhaut_zeta/fomalhaut_test_gap_probs_top10.jsonl'
FOMC='../fomalhaut_zeta/fomalhaut_test_gap_cands.jsonl'
OUT='../../final_solution/submission.csv'

def ajl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)
def chain(shards):
    for p in shards:
        yield from ajl(p)

class Stream:
    def __init__(self,it):
        self.it=it; self.cur=next(it,None)
    def get(self,target):
        while self.cur is not None and self.cur['id']<target:
            self.cur=next(self.it,None)
        if self.cur is not None and self.cur['id']==target:
            o=self.cur; self.cur=next(self.it,None); return o
        return None

def lex(x,w):
    # Levenshtein distance (pure python, small strings)
    if x==w: return 0
    lx=len(x); lw=len(w)
    if lx==0: return lw
    if lw==0: return lx
    prev=list(range(lw+1))
    for i in range(1,lx+1):
        cur=[i]+[0]*lw
        xi=x[i-1]
        for j in range(1,lw+1):
            cur[j]=min(prev[j]+1, cur[j-1]+1, prev[j-1]+(xi!=w[j-1]))
        prev=cur
    return prev[lw]

def fom_mass(o):
    gp=o.get('gap_probs') or []
    if isinstance(gp,dict): gp=list(gp.items())
    return {int(g):float(p) for g,p in gp if float(p)>0}
def norm_cands(o,g):
    d={}
    if o:
        for it in (o.get('cands') or []):
            gg,w,c=int(it[0]),str(it[1]),float(it[2])
            if gg==g and w and c>0: d[w]=d.get(w,0)+c
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}

def medoid(P,mpb=MPB,mean_len=4.57):
    ws=list(P);pool=set(ws)
    for w in ws:
        for k in (1,2,3):
            if len(w)>k:pool.add(w[:k])
    pool|={',','the','a','of','to','and','in','.'}
    best=None
    for x in pool:
        e=sum(P[w]*lex(x,w) for w in ws)+mpb*(len(x)+mean_len)
        if best is None or e<best[0]:best=(e,x)
    return best[1]

def features(tokens, mlmrow, mass):
    n=len(tokens)
    items=sorted(mass.items(),key=lambda kv:-kv[1])[:TOPN]
    topd=dict(items); rest=1.0-sum(topd.values())
    wd=mlmrow.get('words') or {}
    gm=[]
    for gs,pairs0 in wd.items():
        g=int(gs)
        if not(1<=g<n):continue
        pairs=[(w,lp) for w,lp in pairs0 if w]
        if len(pairs)>=2:gm.append((g,pairs))
    if not gm:return None
    missing=[g for g,_ in gm if g not in topd]
    floor=math.log(max(rest,1e-9)/max(1,len(missing))) if missing else -30.0
    rows=[];gaps=[]
    for g,pairs in gm:
        fl=math.log(max(topd[g],1e-12)) if g in topd else floor
        ws=[w for w,_ in pairs];lps=[lp for _,lp in pairs];top1=lps[0];mx=max(lps)
        nxt=tokens[g];prv=tokens[g-1]
        lp_nxt=min(lps) if nxt not in ws else lps[ws.index(nxt)]
        lp_prv=min(lps) if prv not in ws else lps[ws.index(prv)]
        nconf=sum(1 for lp in lps if math.exp(lp)>0.01)
        punct=1.0 if(nxt in PUNCT or prv in PUNCT) else 0.0
        rows.append([fl,top1,mx-lp_nxt,mx-lp_prv,top1-lps[-1],top1-lps[1],g/(n-1),nconf,len(ws[0]),punct])
        gaps.append(g)
    A=np.array(rows);mu=A.mean(0);sd=A.std(0);sd[sd<1e-9]=1.0;Z=(A-mu)/sd
    inter=np.column_stack([Z[:,0]*Z[:,2],Z[:,0]*Z[:,3],Z[:,2]*Z[:,4]])
    Z=np.column_stack([Z,inter])
    return gaps,Z,{g:pairs for g,pairs in gm}

def main():
    z=zipfile.ZipFile('../../data/test_v2.txt.zip')
    rd=csv.reader(io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8'));next(rd)
    mlm=Stream(chain(SHARDS)); fomS=Stream(ajl(FOM)); candS=Stream(ajl(FOMC))
    nact=0;ntot=0
    with open(OUT,'w',encoding='utf-8',newline='') as fo:
        wr=csv.writer(fo,quoting=csv.QUOTE_ALL); wr.writerow(['id','sentence'])
        for r in rd:
            i=int(r[0]); sent=r[1]
            tokens=[t for t in sent.split(' ') if t!='']
            m=mlm.get(i); f=fomS.get(i); c=candS.get(i)
            if m is None or f is None:
                wr.writerow([i,' '.join(tokens)]);ntot+=1;continue
            mass=fom_mass(f)
            b=features(tokens,m,mass)
            if b is None:
                wr.writerow([i,' '.join(tokens)]);ntot+=1;continue
            gaps,Z,gw=b
            s=clf.decision_function(Z); e=np.exp(s-s.max()); P=e/e.sum()
            j=int(np.argmax(P)); g1=gaps[j]; P1=float(P[j])
            if P1<TH:
                wr.writerow([i,' '.join(tokens)])
            else:
                Pw={}
                for w,lp in gw[g1]: Pw[w]=Pw.get(w,0)+math.exp(lp)
                fw=norm_cands(c,g1)
                for w,v in fw.items(): Pw[w]=Pw.get(w,0)+ALPHA*v
                x=medoid(Pw) if Pw else ''
                if x=='':
                    wr.writerow([i,' '.join(tokens)])
                else:
                    wr.writerow([i,' '.join(tokens[:g1]+[x]+tokens[g1:])]);nact+=1
            ntot+=1
            if ntot%25000==0: print('done',ntot,'acted',nact,flush=True)
    print('total',ntot,'acted',nact,flush=True)

if __name__=='__main__':
    main()
