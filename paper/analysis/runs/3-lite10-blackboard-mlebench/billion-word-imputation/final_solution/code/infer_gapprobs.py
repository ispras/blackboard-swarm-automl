import os,sys,json,time,argparse,pickle,csv
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0,'scripts')
from train_detector import Detector

def build_text(tokens):
    return " ".join([w if k==0 else "<mask> "+w for k,w in enumerate(tokens)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--input',required=True,help='csv (test) or txt (gap lines)')
    ap.add_argument('--kind',default='txt',choices=['csv','txt'])
    ap.add_argument('--out',required=True)
    ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--maxlen',type=int,default=256)
    ap.add_argument('--gpu_frac',type=float,default=0.15)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    if args.device=='cuda': torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
    sd=torch.load(args.ckpt,map_location='cpu')
    tok=AutoTokenizer.from_pretrained('roberta-base')
    enc=AutoModel.from_pretrained('roberta-base',add_pooling_layer=False)
    model=Detector(enc,len(sd['words']))
    model.encoder.load_state_dict(sd['encoder']); model.gap_head.load_state_dict(sd['gap_head']); model.word_head.load_state_dict(sd['word_head'])
    model.to(args.device).eval(); mask_id=tok.mask_token_id
    ids=[]; toks=[]
    if args.kind=='csv':
        with open(args.input,newline='') as f:
            r=csv.reader(f); next(r)
            for row in r:
                ids.append(int(row[0])); toks.append(row[1].split())
                if args.limit and len(toks)>=args.limit: break
    else:
        for l in open(args.input):
            toks.append(l.rstrip('\n').split())
            if args.limit and len(toks)>=args.limit: break
    print('sentences',len(toks),flush=True)
    gap_probs=[None]*len(toks); detwp=[None]*len(toks); trunc=[False]*len(toks)
    order=sorted(range(len(toks)),key=lambda i:len(toks[i]))
    t0=time.time()
    with torch.no_grad():
        for s in range(0,len(order),args.bs):
            idxs=order[s:s+args.bs]
            texts=[build_text(toks[i]) for i in idxs]
            encd=tok(texts,padding=True,truncation=True,max_length=args.maxlen,return_tensors='pt')
            ii=encd['input_ids'].to(args.device); am=encd['attention_mask'].to(args.device)
            h=model.encoder(input_ids=ii,attention_mask=am).last_hidden_state
            mp=(ii==mask_id)
            for r,i in enumerate(idxs):
                pos=mp[r].nonzero(as_tuple=True)[0]
                if len(pos)==0:
                    gap_probs[i]=np.zeros(0,np.float32); detwp[i]=np.zeros(0,np.float32); trunc[i]=True; continue
                gl=model.gap_head(h[r,pos]).squeeze(-1).float()
                gp=torch.softmax(gl,dim=0).cpu().numpy().astype(np.float32)
                wl=model.word_head(h[r,pos]).float()
                wp=torch.softmax(wl,dim=-1)
                mx=wp.max(dim=-1).values.cpu().numpy().astype(np.float32)
                gap_probs[i]=gp; detwp[i]=mx; trunc[i]=bool(len(pos)!=max(len(toks[i])-1,0))
            if s%(args.bs*20)==0: print(s,'/',len(order),round(time.time()-t0,1),flush=True)
    pickle.dump({'ids':ids,'tokens':toks,'gap_probs':gap_probs,'detwp':detwp,'trunc':trunc},open(args.out,'wb'),protocol=4)
    print('saved',args.out,'t',round(time.time()-t0,1),'trunc',sum(trunc))

if __name__=='__main__': main()
