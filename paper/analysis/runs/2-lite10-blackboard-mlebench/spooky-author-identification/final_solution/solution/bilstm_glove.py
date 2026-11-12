import numpy as np, pandas as pd, re, time, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
MAXLEN=80
def tok(s): return re.findall(r"[a-z']+", s.lower())
trt=[tok(s) for s in tr['text']]; tet=[tok(s) for s in te['text']]
from collections import Counter
cnt=Counter(w for t in trt for w in t)
vocab=['<pad>','<unk>']+[w for w,c in cnt.most_common() if c>=2]
w2i={w:i for i,w in enumerate(vocab)}; V=len(vocab); print('vocab',V,flush=True)
def to_ids(t,ml=MAXLEN):
    ids=[w2i.get(w,1) for w in t[:ml]]; ids+= [0]*(ml-len(ids)); return ids
tri=np.array([to_ids(t) for t in trt],dtype=np.int64); tei=np.array([to_ids(t) for t in tet],dtype=np.int64)
# glove
D=100; emb=np.zeros((V,D),dtype=np.float32); found=0
with open(f'{OUT}/glove.6B.100d.txt') as f:
    for line in f:
        p=line.rstrip().split(' '); w=p[0]
        if w in w2i:
            emb[w2i[w]]=np.asarray(p[1:],dtype=np.float32); found+=1
print('glove found',found,'/',V,flush=True)
class DS(Dataset):
    def __init__(s,X,yy=None): s.X=X; s.y=yy
    def __len__(s): return len(s.X)
    def __getitem__(s,i):
        d={'x':torch.tensor(s.X[i])}
        if s.y is not None: d['y']=torch.tensor(s.y[i])
        return d
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb=nn.Embedding(V,D,padding_idx=0); s.emb.weight.data.copy_(torch.tensor(emb))
        s.drop=nn.Dropout(0.4)
        s.lstm=nn.LSTM(D,128,num_layers=2,batch_first=True,bidirectional=True,dropout=0.3)
        s.fc=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,3))
    def forward(s,x):
        e=s.drop(s.emb(x))
        o,_=s.lstm(e)
        m=o.max(1).values; a=o.mean(1)
        return s.fc(torch.cat([m,a],-1))
def pred(model,X):
    model.eval(); out=[]
    with torch.no_grad():
        for b in DataLoader(DS(X),batch_size=256):
            out.append(torch.softmax(model(b['x'].cuda()),-1).cpu().numpy())
    return np.concatenate(out)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
oof=np.zeros((len(tr),3)); testp=np.zeros((len(te),3)); t00=time.time()
lsf=nn.CrossEntropyLoss(label_smoothing=0.05)
for k,(itr,iva) in enumerate(folds):
    t0=time.time(); model=Net().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    dl=DataLoader(DS(tri[itr],y[itr]),batch_size=64,shuffle=True,num_workers=2,drop_last=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8)
    best=9;bva=None;bte=None
    for ep in range(8):
        model.train(); tot=0;n=0
        for b in dl:
            opt.zero_grad(); o=model(b['x'].cuda()); loss=lsf(o,b['y'].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            tot+=loss.item()*len(b['y']); n+=len(b['y'])
        sched.step()
        va=pred(model,tri[iva]); s=log_loss(y[iva],va)
        print(f'fold{k} ep{ep} train={tot/n:.4f} val={s:.5f} {time.time()-t0:.0f}s',flush=True)
        if s<best: best=s; bva=va; bte=pred(model,tei)
    oof[iva]=bva; testp+=bte/5
    print(f'fold{k} BEST {best:.5f} {time.time()-t0:.0f}s',flush=True)
    del model; torch.cuda.empty_cache()
sc=log_loss(y,oof); print(f'OOF bilstm_glove: {sc:.5f} total {time.time()-t00:.0f}s',flush=True)
np.save(f'{OUT}/oof_bilstm_glove.npy',oof); np.save(f'{OUT}/test_bilstm_glove.npy',testp)
json.dump({'oof':sc},open(f'{OUT}/result_bilstm_glove.json','w'))
