import sys, os, time, gzip, argparse, csv
import numpy as np
from trigram_model import TrigramLM

def read_inputs(path, is_csv):
    ids=[]; sents=[]
    if is_csv:
        with open(path, newline='', encoding='utf-8') as f:
            r=csv.reader(f); h=next(r)
            for row in r:
                ids.append(row[0]); sents.append(row[1])
    else:
        with open(path, encoding='utf-8') as f:
            for i,line in enumerate(f):
                ids.append(str(i)); sents.append(line.rstrip('\n'))
    return ids, sents

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--csv',action='store_true')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--end',type=int,default=-1)
    ap.add_argument('--topk',type=int,default=2)
    ap.add_argument('--ngram',default='/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram')
    a=ap.parse_args()
    ids,sents=read_inputs(a.input,a.csv)
    if a.end<0 or a.end>len(sents): a.end=len(sents)
    ids=ids[a.start:a.end]; sents=sents[a.start:a.end]
    lm=TrigramLM(out=a.ngram); t0=time.time()
    with gzip.open(a.out,'wt',encoding='utf-8') as fo:
        for i,(sid,s) in enumerate(zip(ids,sents)):
            toks=s.split(' ')
            c=lm.predict_sentence(toks,topk=a.topk)
            # keep per-gap best-gain candidates (predict already returns topk per gap); sort by gap then gain
            for (k,wid,sc,gain) in c:
                fo.write(f'{sid}\t{k}\t{lm.word(wid)}\t{gain:.5f}\t{sc:.5f}\n')
            if (i+1)%20000==0:
                print('done',i+1,'/',len(sents),round(time.time()-t0,1),flush=True)
    print('finished',len(sents),round(time.time()-t0,1))

if __name__=='__main__': main()
