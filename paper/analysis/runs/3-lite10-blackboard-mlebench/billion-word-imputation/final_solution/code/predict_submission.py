import os,sys,json,time,argparse,csv
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from Levenshtein import distance as lev
sys.path.insert(0,'scripts')
from train_detector import Detector
from infer_dump import predict, build_text

def choose_word(words,probs,lam=0.0):
    k=len(words)
    if k==0: return None
    S=sum(probs)
    if S<=0: return words[0]
    pf=[p/S for p in probs]
    best=None;bestu=None
    for u in words:
        e=sum(pf[i]*lev(u,words[i]) for i in range(k))+lam*len(u)
        if best is None or e<best: best=e;bestu=u
    return bestu

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--test',default='/download_job_code/payload/sandbox/data/test_v2.txt')
    ap.add_argument('--out',required=True)
    ap.add_argument('--tau',type=float,default=0.5)
    ap.add_argument('--lam',type=float,default=0.0)
    ap.add_argument('--strategy',default='elev')
    ap.add_argument('--margin',type=float,default=0.0)
    ap.add_argument('--topk',type=int,default=10)
    ap.add_argument('--maxlen',type=int,default=192)
    ap.add_argument('--bs',type=int,default=256)
    ap.add_argument('--chunk',type=int,default=4096)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--gpu_frac',type=float,default=0.0)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.device=='cuda' and args.gpu_frac>0: torch.cuda.set_process_memory_fraction(args.gpu_frac,0) if hasattr(torch.cuda,'set_process_memory_fraction') else torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
    sd=torch.load(args.ckpt,map_location='cpu')
    tok=AutoTokenizer.from_pretrained('roberta-base')
    enc=AutoModel.from_pretrained('roberta-base', add_pooling_layer=False)
    model=Detector(enc,len(sd['words']))
    model.encoder.load_state_dict(sd['encoder']); model.gap_head.load_state_dict(sd['gap_head']); model.word_head.load_state_dict(sd['word_head'])
    model.words=sd['words']; model.to(args.device)
    # read test csv
    ids=[]; sents=[]
    with open(args.test,newline='') as f:
        r=csv.reader(f); h=next(r)
        for row in r:
            ids.append(int(row[0])); sents.append(row[1])
            if args.limit and len(ids)>=args.limit: break
    print('loaded',len(sents),flush=True)
    with open(args.out,'w',newline='') as f:
        w=csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(['id','sentence'])
        t0=time.time(); napp=0
        for s in range(0,len(sents),args.chunk):
            chunk=sents[s:s+args.chunk]
            toks=[x.split() for x in chunk]
            res=predict(model,tok,toks,topk=args.topk,maxlen=args.maxlen,bs=args.bs,device=args.device)
            for j,r in enumerate(res):
                t=toks[j]; gp=r['gap_probs']
                pred=' '.join(t)
                if len(gp)>0 and not r['trunc']:
                    gs=int(gp.argmax()); pb=float(gp[gs])
                    sec=float(np.sort(gp)[-2]) if len(gp)>1 else 0.0
                    if pb>=args.tau and (pb-sec)>=args.margin:
                        words=r['words'][gs]; probs=r['probs'][gs]
                        if args.strategy=='elev': u=choose_word(words,probs,args.lam)
                        else: u=words[0]
                        if u: pred=' '.join(t[:gs+1])+' '+u+' '+' '.join(t[gs+1:]); napp+=1
                w.writerow([ids[s+j],pred])
            print(f'{min(s+args.chunk,len(sents))}/{len(sents)} applied_so_far={napp} elapsed={time.time()-t0:.1f}s',flush=True)
    print('done applied',napp,'/',len(sents))

if __name__=='__main__': main()
