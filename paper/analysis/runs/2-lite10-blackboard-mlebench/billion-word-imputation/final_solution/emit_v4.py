"""Final assembler v2/v4.
Usage: python emit_v4.py <gaps.jsonl|tsv> <lm.jsonl> <out.csv> [THR] [trigram.jsonl]
gaps: JSONL {id,gap,margin,...} OR TSV id<TAB>gap<TAB>margin
lm:   {id, base:[[w,lp]..], large:[[w,lp]..]}
trigram (optional): {id, gap, tri:[[word,count]..]}  (words at chosen gap)
Fallback fomalhaut old top3 cands: ../fomalhaut_zeta/fomalhaut_test_gap_cands.jsonl
Recipe: P = 0.3*base + 0.7*large + 0.4*trigram(normalized); medoid mpb=0.05 depth=3.
"""
import csv, io, json, math, sys, zipfile
import Levenshtein
GAPS=sys.argv[1]; LMF=sys.argv[2]; OUTF=sys.argv[3]
THR=float(sys.argv[4]) if len(sys.argv)>4 else 0.35
TRIF=sys.argv[5] if len(sys.argv)>5 else None
MPB=float(sys.argv[6]) if len(sys.argv)>6 else 0.05
ALPHA=float(sys.argv[7]) if len(sys.argv)>7 else 0.4
OLDF='../fomalhaut_zeta/fomalhaut_test_gap_cands.jsonl'
def read_gaps(p):
    d={}
    if p.endswith('.tsv'):
        for l in open(p):
            x=l.split()
            if len(x)>=3: d[int(x[0])]=(int(x[1]),float(x[2]))
            elif len(x)==2: d[int(x[0])]=(int(x[1]),1.0)
    else:
        for l in open(p):
            o=json.loads(l); d[int(o['id'])]=(int(o['gap']),float(o.get('margin',1.0)))
    return d
gaps=read_gaps(GAPS); print('gaps',len(gaps),flush=True)
lm={}
for l in open(LMF):
    o=json.loads(l); lm[int(o['id'])]=(o['base'],o['large'])
print('lm',len(lm),flush=True)
tri={}
if TRIF:
    for l in open(TRIF):
        o=json.loads(l); tri[int(o['id'])]=o
    print('tri',len(tri),flush=True)
def ajl(p):
    with open(p) as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)
def norm_counts(pairs):
    d={}
    for w,c in pairs:
        if w and c>0: d[w]=d.get(w,0)+float(c)
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}
def norm_cands(o,g):
    d={}
    if o:
        for it in (o.get('cands') or []):
            gg,w,c=int(it[0]),str(it[1]),float(it[2])
            if gg==g and w and c>0: d[w]=d.get(w,0)+c
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}
def medoid(P,mpb=None,depth=3,ml=4.57):
    if mpb is None: mpb=MPB
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
    old=ajl(OLDF); ocur=next(old,None)
    z=zipfile.ZipFile('../../data/test_v2.txt.zip')
    rd=csv.reader(io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8'));next(rd)
    n=0;na=0
    with open(OUTF,'w',encoding='utf-8',newline='') as fo:
        wr=csv.writer(fo,quoting=csv.QUOTE_ALL);wr.writerow(['id','sentence'])
        for r in rd:
            i=int(r[0]);T=[t for t in r[1].split(' ') if t]
            while ocur is not None and ocur['id']<i: ocur=next(old,None)
            oc=ocur if (ocur is not None and ocur['id']==i) else None
            if i not in gaps or gaps[i][1]<THR or i not in lm:
                wr.writerow([i,' '.join(T)]);n+=1;continue
            g=min(gaps[i][0],len(T))
            base,large=lm[i]
            P={}
            for w,lp in base[:15]:
                if w:P[w]=P.get(w,0)+0.3*math.exp(lp)
            for w,lp in large[:15]:
                if w:P[w]=P.get(w,0)+0.7*math.exp(lp)
            if i in tri and int(tri[i].get('gap',-1))==g:
                for w,v in norm_counts(tri[i].get('tri') or []).items(): P[w]=P.get(w,0)+ALPHA*v
            else:
                for w,v in norm_cands(oc,g).items(): P[w]=P.get(w,0)+ALPHA*v
            x=medoid(P) if P else ''
            if x=='': wr.writerow([i,' '.join(T)])
            else: wr.writerow([i,' '.join(T[:g]+[x]+T[g:])]);na+=1
            n+=1
            if n%50000==0: print(n,na,flush=True)
    print('total',n,'acted',na,flush=True)
if __name__=='__main__': main()
