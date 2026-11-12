import argparse, csv, io, json, sys, zipfile, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
def norm(t): return t.replace('\u0120',' ').replace('\u010a',' ').replace('\u0109',' ').strip()
ap=argparse.ArgumentParser()
ap.add_argument('--gaps',required=True); ap.add_argument('--src',required=True,choices=['test','val'])
ap.add_argument('--out',required=True); ap.add_argument('--batch',type=int,default=128)
ap.add_argument('--maxlen',type=int,default=192); ap.add_argument('--topk',type=int,default=20)
a=ap.parse_args()
sents={}
if a.src=='test':
    z=zipfile.ZipFile('../../data/test_v2.txt.zip'); rd=csv.reader(io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8'));next(rd)
    for r in rd: sents[int(r[0])]=r[1]
else:
    for r in csv.DictReader(open('val_test.csv',encoding='utf-8')): sents[int(r['id'])]=r['sentence']
gaps=[]
for line in open(a.gaps):
    p=line.split()
    if len(p)==2: gaps.append((int(p[0]),int(p[1])))
print('gaps',len(gaps),flush=True)
def run(name):
    tok=AutoTokenizer.from_pretrained(name,add_prefix_space=True)
    model=AutoModelForMaskedLM.from_pretrained(name,torch_dtype=torch.float16).eval().cuda()
    cache={}; out={}
    items=[]
    for di,(i,g) in enumerate(gaps):
        words=[w for w in sents[i].split(' ') if w]
        key=(id(words),len(words))  # not reliable; just re-tokenize per item (cache by i)
        enc=cache.get(i)
        if enc is None:
            enc=tok(words,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=a.maxlen)
            cache[i]=(enc['input_ids'],enc.word_ids())
        ids,wids=cache[i]
        first={}
        for ti,w in enumerate(wids):
            if w is not None and w not in first: first[w]=ti
        if g not in first: continue
        items.append((di,i,g,ids,first[g]))
    with torch.no_grad():
        for bi in range(0,len(items),a.batch):
            b=items[bi:bi+a.batch]; ml=max(len(x[3])+1 for x in b)
            inp=torch.full((len(b),ml),tok.pad_token_id,dtype=torch.long);att=torch.zeros((len(b),ml),dtype=torch.long);posl=[]
            for j,(di,i,g,ids,pos) in enumerate(b):
                seq=ids[:pos]+[tok.mask_token_id]+ids[pos:];L=len(seq)
                inp[j,:L]=torch.tensor(seq);att[j,:L]=1;posl.append(pos)
            inp=inp.cuda();att=att.cuda()
            lo=model(input_ids=inp,attention_mask=att).logits
            idx=torch.arange(len(b),device='cuda')
            lp=torch.log_softmax(lo[idx,torch.tensor(posl,device='cuda')].float(),dim=-1)
            tv,ti=lp.topk(a.topk,dim=-1);tv=tv.cpu().numpy();ti=ti.cpu().numpy()
            for j,(di,i,g,ids,pos) in enumerate(b):
                ws=[]
                for t,v in zip(ti[j],tv[j]):
                    x=norm(tok.convert_ids_to_tokens(int(t)))
                    if x: ws.append([x,float(v)])
                out[i]=ws
            del lo
            if bi%(a.batch*20)==0: print(name,bi,'/',len(items),flush=True)
    del model; torch.cuda.empty_cache()
    return out
base=run('roberta-base'); large=run('roberta-large')
with open(a.out,'w') as f:
    for i,g in gaps:
        f.write(json.dumps({'id':i,'gap':g,'base':base.get(i,[]),'large':large.get(i,[])})+'\n')
print('wrote',a.out,flush=True)
