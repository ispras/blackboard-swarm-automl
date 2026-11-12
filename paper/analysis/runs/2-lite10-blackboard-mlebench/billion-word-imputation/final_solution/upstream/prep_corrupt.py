import argparse, random, numpy as np
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='train_v2.txt')
    ap.add_argument('--model',default='roberta-base')
    ap.add_argument('--out',default='corrupt_train.npz')
    ap.add_argument('--n',type=int,default=500000)
    ap.add_argument('--max_line',type=int,default=18000000)
    ap.add_argument('--stride',type=int,default=36)
    ap.add_argument('--maxlen',type=int,default=60)
    ap.add_argument('--maxtok',type=int,default=96)
    ap.add_argument('--maxw',type=int,default=64)
    ap.add_argument('--seed',type=int,default=23)
    args=ap.parse_args()
    rng=random.Random(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model, **({"add_prefix_space":True} if ("roberta" in args.model or "gpt2" in args.model) else {}))
    ids_m=np.full((args.n,args.maxtok),1,dtype=np.int32)
    ws_m=np.full((args.n,args.maxw),-1,dtype=np.int32)
    nw_m=np.zeros(args.n,dtype=np.int32); tg_m=np.zeros(args.n,dtype=np.int32)
    ln_m=np.zeros(args.n,dtype=np.int32)
    row=0
    with open(args.train,encoding='utf-8',errors='replace') as f:
        for i,line in enumerate(f):
            if i>=args.max_line or row>=args.n: break
            if i%args.stride: continue
            words=line.strip().split(' ')
            if not (5<=len(words)<=args.maxlen): continue
            tg=rng.randint(1,len(words)-2)
            C=words[:tg]+words[tg+1:]
            enc=tok(C,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=args.maxtok)
            ids=enc['input_ids']; wids=enc.word_ids()
            first={}
            for ti,w in enumerate(wids):
                if w is not None and w not in first: first[w]=ti
            if len(first)!=len(C): continue
            n=len(C)
            ids_m[row,:len(ids)]=ids
            ws_m[row,:n]=[first[g] for g in range(n)]
            nw_m[row]=n; tg_m[row]=tg; ln_m[row]=len(ids)
            row+=1
    np.savez(args.out,ids=ids_m[:row],wstart=ws_m[:row],nw=nw_m[:row],tg=tg_m[:row],ln=ln_m[:row])
    print('rows',row)

if __name__=='__main__': main()
