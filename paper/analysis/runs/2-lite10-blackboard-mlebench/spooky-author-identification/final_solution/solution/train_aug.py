import os, sys, time, json, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
p=argparse.ArgumentParser()
p.add_argument('--model',default='microsoft/deberta-v3-large'); p.add_argument('--tag',default='dbl5')
p.add_argument('--maxlen',type=int,default=128); p.add_argument('--bs',type=int,default=16)
p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--epochs',type=int,default=2)
p.add_argument('--ls',type=float,default=0.1); p.add_argument('--wd',type=float,default=0.01)
p.add_argument('--warmup',type=float,default=0.1); p.add_argument('--seed',type=int,default=42)
p.add_argument('--evalbs',type=int,default=64)
p.add_argument('--maskp',type=float,default=0.0)
a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
def enc(t):
    e=tok(list(t),truncation=True,max_length=a.maxlen,padding='max_length',return_tensors='np'); return e['input_ids'],e['attention_mask']
SPECIAL=set(tok.all_special_ids)
class DS(Dataset):
    def __init__(s,i,m,yy=None,maskp=0.0): s.i=i; s.m=m; s.y=yy; s.maskp=maskp
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        ids=torch.tensor(s.i[k]).clone(); mask=torch.tensor(s.m[k])
        if s.maskp>0 and s.y is not None:
            r=torch.rand(ids.shape)
            mm=(r<s.maskp)
            for sp in SPECIAL: mm &= (ids!=sp)
            ids[mm]=tok.mask_token_id
        d={'input_ids':ids,'attention_mask':mask}
        if s.y is not None: d['labels']=torch.tensor(s.y[k])
        return d
tri,trm=enc(tr['text'].values); tei,tem=enc(te['text'].values)
def pred(model,ids,mask):
    model.eval(); out=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for b in DataLoader(DS(ids,mask),batch_size=a.evalbs):
            o=model(input_ids=b['input_ids'].to('cuda'),attention_mask=b['attention_mask'].to('cuda'))
            out.append(torch.softmax(o.logits.float(),-1).cpu().numpy())
    return np.concatenate(out)
lsf=torch.nn.CrossEntropyLoss(label_smoothing=a.ls)
oof=np.zeros((len(tr),3)); testp=np.zeros((len(te),3)); t00=time.time()
for k,(itr,iva) in enumerate(folds):
    t0=time.time()
    model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=3).to('cuda')
    dl=DataLoader(DS(tri[itr],trm[itr],y[itr],maskp=a.maskp),batch_size=a.bs,shuffle=True,num_workers=4,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd)
    steps=len(dl)*a.epochs; sch=get_linear_schedule_with_warmup(opt,int(a.warmup*steps),steps)
    best=9; bva=None; bte=None
    for ep in range(a.epochs):
        model.train(); tot=0;n=0
        for b in dl:
            opt.zero_grad()
            with torch.autocast('cuda',dtype=torch.float16):
                o=model(input_ids=b['input_ids'].to('cuda'),attention_mask=b['attention_mask'].to('cuda'))
                loss=lsf(o.logits.float(),b['labels'].to('cuda'))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
            tot+=loss.item()*len(b['labels']); n+=len(b['labels'])
        va=pred(model,tri[iva],trm[iva]); s=log_loss(y[iva],va)
        print(f'fold{k} ep{ep} train={tot/n:.4f} val={s:.5f} {time.time()-t0:.0f}s',flush=True)
        if s<best: best=s; bva=va; bte=pred(model,tei,tem)
    oof[iva]=bva; testp+=bte/len(folds)
    print(f'fold{k} BEST {best:.5f} {time.time()-t0:.0f}s',flush=True)
    np.save(f'{OUT}/oof_{a.tag}_partial.npy',oof); np.save(f'{OUT}/test_{a.tag}_partial.npy',testp*len(folds)/(k+1))
    del model; torch.cuda.empty_cache()
sc=log_loss(y,oof); print(f'OOF {a.tag}: {sc:.5f} total {time.time()-t00:.0f}s',flush=True)
np.save(f'{OUT}/oof_{a.tag}.npy',oof); np.save(f'{OUT}/test_{a.tag}.npy',testp)
json.dump({'tag':a.tag,'model':a.model,'oof':sc,'args':vars(a)},open(f'{OUT}/result_{a.tag}.json','w'),indent=2)
