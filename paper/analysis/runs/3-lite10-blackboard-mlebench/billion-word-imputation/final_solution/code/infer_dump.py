import os,sys,json,time,argparse,pickle
import numpy as np, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from Levenshtein import distance as lev
sys.path.insert(0,'scripts')
from train_detector import Detector

def build_text(tokens):
    return " ".join([w if k==0 else "<mask> "+w for k,w in enumerate(tokens)])

@torch.no_grad()
def predict(model,tok,tokens_list,topk=10,maxlen=192,bs=256,device='cuda'):
    model.eval()
    mask_id=tok.mask_token_id
    res=[None]*len(tokens_list)
    order=sorted(range(len(tokens_list)),key=lambda i:len(tokens_list[i]))
    for s in range(0,len(order),bs):
        idxs=order[s:s+bs]
        texts=[build_text(tokens_list[i]) for i in idxs]
        enc=tok(texts,padding=True,truncation=True,max_length=maxlen,return_tensors='pt')
        ids=enc['input_ids'].to(device); attn=enc['attention_mask'].to(device)
        h=model.encoder(input_ids=ids,attention_mask=attn).last_hidden_state
        mp=(ids==mask_id)
        for r,i in enumerate(idxs):
            pos=mp[r].nonzero(as_tuple=True)[0]
            if len(pos)==0:
                res[i]={'gap_probs':np.zeros(0,np.float32),'words':[],'probs':[],'trunc':True}
                continue
            gl=model.gap_head(h[r,pos]).squeeze(-1).float()
            gp=torch.softmax(gl,dim=0).cpu().numpy()
            wl=model.word_head(h[r,pos]).float()
            wp=torch.softmax(wl,dim=-1)
            k=min(topk,wp.size(1))
            tv,ti=torch.topk(wp,k,dim=-1)
            tv=tv.cpu().numpy(); ti=ti.cpu().numpy()
            words=[[model.words[int(j)] for j in ti[g]] for g in range(len(pos))]
            probs=[[float(x) for x in tv[g]] for g in range(len(pos))]
            res[i]={'gap_probs':gp.astype(np.float32),'words':words,'probs':probs,
                    'trunc': bool(mp[r].sum().item()!=max(len(tokens_list[i])-1,0))}
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--meta',default='')
    ap.add_argument('--orig',default='')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--topk',type=int,default=10)
    ap.add_argument('--maxlen',type=int,default=192)
    ap.add_argument('--bs',type=int,default=256)
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--gpu_frac',type=float,default=0.0)
    args=ap.parse_args()
    if args.device=='cuda' and args.gpu_frac>0: torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
    sd=torch.load(args.ckpt,map_location='cpu')
    tok=AutoTokenizer.from_pretrained('roberta-base')
    enc=AutoModel.from_pretrained('roberta-base', add_pooling_layer=False)
    model=Detector(enc,len(sd['words']))
    model.encoder.load_state_dict(sd['encoder']); model.gap_head.load_state_dict(sd['gap_head']); model.word_head.load_state_dict(sd['word_head'])
    model.words=sd['words']; model.to(args.device)
    toks=[l.rstrip('\n').split() for l in open(args.input)]
    if args.limit: toks=toks[:args.limit]
    meta=[]; orig=[]
    if args.meta:
        rows=[l.rstrip('\n').split('\t') for l in open(args.meta)][1:]
        if args.limit: rows=rows[:args.limit]
        meta=rows
    if args.orig:
        orig=[l.rstrip('\n') for l in open(args.orig)]
        if args.limit: orig=orig[:args.limit]
    t0=time.time()
    res=predict(model,tok,toks,topk=args.topk,maxlen=args.maxlen,bs=args.bs,device=args.device)
    print('infer done',time.time()-t0,'trunc',sum(r['trunc'] for r in res if r),flush=True)
    pickle.dump({'tokens':toks,'res':res,'meta':meta,'orig':orig},open(args.out,'wb'),protocol=4)
    print('saved',args.out)

if __name__=='__main__': main()
