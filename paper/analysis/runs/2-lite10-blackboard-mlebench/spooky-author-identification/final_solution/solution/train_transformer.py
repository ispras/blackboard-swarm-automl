import os, sys, time, json, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

p=argparse.ArgumentParser()
p.add_argument('--model',default='roberta-base'); p.add_argument('--tag',default='roberta')
p.add_argument('--maxlen',type=int,default=160); p.add_argument('--bs',type=int,default=32)
p.add_argument('--lr',type=float,default=2e-5); p.add_argument('--epochs',type=int,default=4)
p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=42)
p.add_argument('--maxtrain',type=int,default=0)
a=p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
skf=StratifiedKFold(a.folds,shuffle=True,random_state=42)
if a.maxtrain:
    idx=np.random.RandomState(0).choice(len(tr),a.maxtrain,replace=False); tr=tr.iloc[idx].reset_index(drop=True); te=te; y=y[idx]
    skf=StratifiedKFold(a.folds,shuffle=True,random_state=42)
folds=list(skf.split(tr,y))
tok=AutoTokenizer.from_pretrained(a.model)
def enc(texts):
    e=tok(list(texts),truncation=True,max_length=a.maxlen,padding='max_length',return_tensors='np')
    return e['input_ids'],e['attention_mask']
class DS(Dataset):
    def __init__(s,ids,mask,yy=None): s.ids=ids; s.mask=mask; s.y=yy
    def __len__(s): return len(s.ids)
    def __getitem__(s,i):
        d={'input_ids':torch.tensor(s.ids[i]),'attention_mask':torch.tensor(s.mask[i])}
        if s.y is not None: d['labels']=torch.tensor(s.y[i])
        return d
tri,trm=enc(tr['text']); tei,tem=enc(te['text'])
dev='cuda'
def predict(model,ids,mask,bs=64):
    model.eval(); ds=DS(ids,mask); dl=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=2)
    out=[]
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        for b in dl:
            o=model(input_ids=b['input_ids'].to(dev),attention_mask=b['attention_mask'].to(dev))
            out.append(torch.softmax(o.logits.float(),-1).cpu().numpy())
    return np.concatenate(out)
oof=np.zeros((len(tr),3)); testp=np.zeros((len(te),3)); t00=time.time()
for k,(itr,iva) in enumerate(folds):
    t0=time.time()
    model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=3).to(dev)
    dtr=DS(tri[itr],trm[itr],y[itr]); dl=DataLoader(dtr,batch_size=a.bs,shuffle=True,num_workers=4,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    steps=len(dl)*a.epochs; sch=get_linear_schedule_with_warmup(opt,int(0.1*steps),steps)
    best=9e9; bestva=None; bestte=None
    for ep in range(a.epochs):
        model.train(); tl=0; n=0
        for b in dl:
            opt.zero_grad()
            with torch.autocast('cuda',dtype=torch.float16):
                o=model(input_ids=b['input_ids'].to(dev),attention_mask=b['attention_mask'].to(dev),labels=b['labels'].to(dev))
            o.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); tl+=o.loss.item()*len(b['labels']); n+=len(b['labels'])
        va=predict(model,tri[iva],trm[iva]); s=log_loss(y[iva],va)
        print(f'fold{k} ep{ep} trainloss {tl/n:.4f} val {s:.5f} {time.time()-t0:.0f}s',flush=True)
        if s<best: best=s; bestva=va; bestte=predict(model,tei,tem)
    oof[iva]=bestva; testp+=bestte/len(folds)
    print(f'fold{k} best {best:.5f} total {time.time()-t0:.0f}s',flush=True)
    del model; torch.cuda.empty_cache()
sc=log_loss(y,oof); print(f'OOF {a.tag}: {sc:.5f} total {time.time()-t00:.0f}s',flush=True)
np.save(f'{OUT}/oof_{a.tag}.npy',oof); np.save(f'{OUT}/test_{a.tag}.npy',testp)
json.dump({'tag':a.tag,'model':a.model,'oof':sc,'args':vars(a)},open(f'{OUT}/result_{a.tag}.json','w'),indent=2)
