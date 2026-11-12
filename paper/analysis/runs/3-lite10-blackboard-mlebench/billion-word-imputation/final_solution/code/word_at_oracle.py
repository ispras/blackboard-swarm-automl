import pickle,sys,argparse,time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def build_vocab(tok):
    V=tok.vocab_size; ids=[]; id2w={}
    # detect word-start marker
    sample=[tok.convert_ids_to_tokens(i) for i in range(min(V,5000))]
    marker=None
    if any(isinstance(s,str) and s.startswith('Ġ') for s in sample): marker='Ġ'
    elif any(isinstance(s,str) and s.startswith('▁') for s in sample): marker='▁'
    special=set(tok.all_special_ids)
    for i in range(V):
        if i in special: continue
        s=tok.convert_ids_to_tokens(i)
        if s is None: continue
        if marker is not None:
            if not s.startswith(marker): continue
            w=tok.convert_tokens_to_string([s]).strip()
        else:
            if s.startswith('##'): continue
            w=tok.convert_tokens_to_string([s]).strip()
        if w=='' or ' ' in w: continue
        id2w[i]=w; ids.append(i)
    return np.array(ids,dtype=np.int64), id2w

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True); ap.add_argument('--dump',required=True)
    ap.add_argument('--mode',default='oracle'); ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--bs',type=int,default=64); ap.add_argument('--maxlen',type=int,default=160)
    ap.add_argument('--out',required=True); ap.add_argument('--gpu_frac',type=float,default=0.15)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    if args.device=='cuda': torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
    D=pickle.load(open(args.dump,'rb')); toks=D['tokens']; res=D['res']; meta=D['meta']
    tok=AutoTokenizer.from_pretrained(args.ckpt); model=AutoModelForMaskedLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).to(args.device).eval()
    ids_all,id2w=build_vocab(tok); ids_t=torch.tensor(ids_all,device=args.device)
    texts=[];keys=[]
    for i in range(len(toks)):
        g=int(meta[i][2]) if args.mode=='oracle' else (int(res[i]['gap_probs'].argmax())+1 if len(res[i]['gap_probs']) else -1)
        t=toks[i]
        if g<1 or g>len(t)-1: continue
        texts.append(' '.join(t[:g]+[tok.mask_token]+t[g:])); keys.append(i)
    print('n',len(texts),flush=True); out={}; t0=time.time()
    for s in range(0,len(texts),args.bs):
        chunk=texts[s:s+args.bs]
        enc=tok(chunk,padding=True,truncation=True,max_length=args.maxlen,return_tensors='pt')
        ii=enc['input_ids'].to(args.device); am=enc['attention_mask'].to(args.device)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            lo=model(input_ids=ii,attention_mask=am).logits
        mp=(ii==tok.mask_token_id); idx=mp.float().argmax(1); ar=torch.arange(len(chunk),device=args.device)
        pr=torch.softmax(torch.log_softmax(lo[ar,idx].float(),-1)[:,ids_t],-1)
        k=min(args.topk,pr.size(1)); tv,ti=torch.topk(pr,k,-1); tv=tv.cpu().numpy(); ti=ti.cpu().numpy()
        for r in range(len(chunk)):
            out[keys[s+r]]=([id2w[int(ids_all[ti[r,j]])] for j in range(k)],[float(tv[r,j]) for j in range(k)])
        if s%(args.bs*20)==0: print(s,'/',len(texts),round(time.time()-t0,1),flush=True)
    pickle.dump(out,open(args.out,'wb'),protocol=4); print('saved',args.out,round(time.time()-t0,1))

if __name__=='__main__': main()
