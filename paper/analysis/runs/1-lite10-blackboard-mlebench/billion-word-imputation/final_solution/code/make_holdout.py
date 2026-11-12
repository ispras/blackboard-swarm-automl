import csv, random, sys, os

TRAIN='train_v2.txt'
N_TOTAL=29998384
N=20000
SEED=12345

def main():
    rng=random.Random(SEED)
    idxs=sorted(rng.sample(range(N_TOTAL), N))
    want={i:k for k,i in enumerate(idxs)}
    out=[]
    with open(TRAIN, encoding='utf-8') as f:
        for ln,line in enumerate(f):
            if ln not in want: 
                if ln>idxs[-1]: break
                continue
            s=line.rstrip('\n')
            toks=s.split()
            if len(toks)<4: continue
            p=rng.randrange(1,len(toks)-1)
            deleted=toks[p]
            masked=' '.join(toks[:p]+toks[p+1:])
            out.append((ln,p,deleted,s,masked))
    with open('holdout_20k.tsv','w',encoding='utf-8',newline='') as g:
        w=csv.writer(g,delimiter='\t')
        w.writerow(['line_idx','pos','deleted','original','masked'])
        for row in out: w.writerow(row)
    print('wrote',len(out),'rows')

if __name__=='__main__':
    main()
