"""Apply a decisions file (id<TAB>gap<TAB>word) to test_v2.txt and write submission.csv.
Validates ids, gap bounds, token sanity. Missing ids / gap=-1 => leave sentence unchanged.
"""
import csv, sys, argparse, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--test',default='/download_job_code/payload/sandbox/workspaces/antares_upsilon/test_v2.txt')
    ap.add_argument('--decisions',required=True,help='tsv id<TAB>gap<TAB>word (gap<0 means skip)')
    ap.add_argument('--out',required=True)
    ap.add_argument('--stats',default=None)
    a=ap.parse_args()
    dec={}
    with open(a.decisions,encoding='utf-8') as f:
        for ln,line in enumerate(f):
            line=line.rstrip('\n')
            if not line: continue
            parts=line.split('\t')
            if len(parts)<3: raise ValueError(f'bad decisions line {ln}: {line!r}')
            i=int(parts[0]); k=int(parts[1]); w=parts[2]
            if k>=0:
                if ' ' in w or '\t' in w or w=='':
                    raise ValueError(f'bad word at line {ln}: {w!r}')
                dec[i]=(k,w)
    n=0; applied=0; ids=set()
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    with open(a.test,newline='',encoding='utf-8') as fi, open(a.out,'w',newline='',encoding='utf-8') as fo:
        r=csv.reader(fi); h=next(r)
        wr=csv.writer(fo,quoting=csv.QUOTE_NONNUMERIC,lineterminator='\n')
        wr.writerow(['id','sentence'])
        for row in r:
            i=int(row[0]); s=row[1]; ids.add(i)
            if i in dec:
                k,w=dec[i]; toks=s.split(' ')
                if not (1<=k<=len(toks)-1):
                    raise ValueError(f'gap {k} out of range for id {i} (m={len(toks)})')
                s=' '.join(toks[:k]+[w]+toks[k:]); applied+=1
            wr.writerow([i,s]); n+=1
    assert n==len(ids)==302644, (n,len(ids))
    assert min(ids)==0 and max(ids)==302643
    extra=set(dec)-ids
    assert not extra, f'{len(extra)} decision ids not in test, e.g. {list(extra)[:5]}'
    msg=f'wrote {a.out}: rows={n} applied={applied} coverage={applied/n:.4f}'
    print(msg)
    if a.stats:
        with open(a.stats,'w') as f: f.write(msg+'\n')
if __name__=='__main__': main()
