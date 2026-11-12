"""Emit submission using final fused gap file + base+large+fomalhaut medoid recipe.
Usage: python emit_v3_recipe.py <gaps.jsonl> <test_lm_chosen.jsonl> <out.csv> [THR]
gaps.jsonl: {"id","gap","margin"} (fomalhaut v3)."""
import csv, io, json, math, sys, zipfile
import Levenshtein
GAPS=sys.argv[1]; LMF=sys.argv[2]; OUTF=sys.argv[3]; THR=float(sys.argv[4]) if len(sys.argv)>4 else 0.30
FOM='../fomalhaut_zeta/fomalhaut_test_gap_cands.jsonl'
gaps={}
for l in open(GAPS,encoding='utf-8'):
    o=json.loads(l); gaps[int(o['id'])]=(int(o['gap']),float(o['margin']))
print('gaps',len(gaps),flush=True)
lm={}
for l in open(LMF,encoding='utf-8'):
    o=json.loads(l); lm[int(o['id'])]=(o['base'],o['large'])
print('lm',len(lm),flush=True)
def ajl(p):
    with open(p,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)
def norm_cands(o,g):
    d={}
    if o:
        for it in (o.get('cands') or []):
            gg,w,c=int(it[0]),str(it[1]),float(it[2])
            if gg==g and w and c>0: d[w]=d.get(w,0)+c
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}
def medoid(P,mpb=0.05,depth=3,ml=4.57):
    ws=list(P);pool=set(ws)
    for w in ws:
        for k in range(1,depth+1):
            if len(w)>k:pool.add(w[:k])
    pool|={',','the','a','of','to','and','in','.'}
    best=None
    for x in pool:
        e=sum(P[w]*Levenshtein.distance(x,w) for w in ws)+mpb*(len(x)+ml)
        if best is None or e<best[0]:best=(e,x)
    return best[1]
def main():
    fom=ajl(FOM); fcur=next(fom,None)
    z=zipfile.ZipFile('../../data/test_v2.txt.zip')
    rd=csv.reader(io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8'));next(rd)
    n=0;na=0
    with open(OUTF,'w',encoding='utf-8',newline='') as fo:
        wr=csv.writer(fo,quoting=csv.QUOTE_ALL);wr.writerow(['id','sentence'])
        for r in rd:
            i=int(r[0]);T=[t for t in r[1].split(' ') if t]
            while fcur is not None and fcur['id']<i: fcur=next(fom,None)
            fc=fcur if (fcur is not None and fcur['id']==i) else None
            if i not in gaps or gaps[i][1]<THR or i not in lm:
                wr.writerow([i,' '.join(T)]);n+=1;continue
            g=min(gaps[i][0],len(T))
            base,large=lm[i]
            P={}
            for w,lp in base[:15]:
                if w:P[w]=P.get(w,0)+0.3*math.exp(lp)
            for w,lp in large[:15]:
                if w:P[w]=P.get(w,0)+0.7*math.exp(lp)
            for w,v in norm_cands(fc,g).items():P[w]=P.get(w,0)+0.4*v
            x=medoid(P) if P else ''
            if x=='': wr.writerow([i,' '.join(T)])
            else: wr.writerow([i,' '.join(T[:g]+[x]+T[g:])]);na+=1
            n+=1
            if n%50000==0: print(n,na,flush=True)
    print('total',n,'acted',na,flush=True)
if __name__=='__main__': main()
