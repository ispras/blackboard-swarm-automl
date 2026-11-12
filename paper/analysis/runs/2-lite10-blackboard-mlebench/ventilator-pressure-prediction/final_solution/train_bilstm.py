import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, argparse, time, os, json

p = argparse.ArgumentParser()
p.add_argument('--model', default='lstm')
p.add_argument('--fold', type=int, default=0)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--bs', type=int, default=512)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--hidden', type=int, default=512)
p.add_argument('--layers', type=int, default=2)
p.add_argument('--dmodel', type=int, default=256)
p.add_argument('--tlayer', type=int, default=4)
p.add_argument('--heads', type=int, default=8)
p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--exp_w', type=float, default=0.0)
p.add_argument('--tag', default='')
p.add_argument('--cache', default='cache')

class ConformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(Fdim, a.dmodel)
        self.pos = nn.Parameter(torch.randn(1,80,a.dmodel)*0.02)
        self.conv = nn.Sequential(
            nn.Conv1d(a.dmodel, a.dmodel, 3, padding=1), nn.GELU(),
            nn.Conv1d(a.dmodel, a.dmodel, 3, padding=1), nn.GELU())
        layer = nn.TransformerEncoderLayer(a.dmodel, a.heads, dim_feedforward=4*a.dmodel,
                                           dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, a.tlayer)
        self.norm = nn.LayerNorm(a.dmodel)
        self.head = nn.Sequential(nn.Linear(a.dmodel, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, x):
        h = self.inp(x) + self.pos
        h = h + self.conv(h.transpose(1,2)).transpose(1,2)
        h = self.enc(h)
        return self.head(self.norm(h)).squeeze(-1)

a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(4)
dev = 'cuda'
os.makedirs('preds', exist_ok=True)
os.makedirs('models', exist_ok=True)

d = np.load(f'{a.cache}/train.npz'); dt = np.load(f'{a.cache}/test.npz')
X = torch.tensor(d['X']); y = torch.tensor(d['y']); uo = torch.tensor(d['u_out'])
folds = np.load(f'{a.cache}/folds.npy')
Xte = torch.tensor(dt['X']); ute = torch.tensor(dt['u_out'])
Fdim = X.shape[2]
tr_idx = np.where(folds != a.fold)[0]; va_idx = np.where(folds == a.fold)[0]
print('train', len(tr_idx), 'val', len(va_idx), 'F', Fdim, flush=True)

Xg = X.to(dev); yg = y.to(dev); uog = uo.to(dev)
Xteg = Xte.to(dev)

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(Fdim, a.hidden), nn.LayerNorm(a.hidden), nn.GELU())
        self.lstm = nn.LSTM(a.hidden, a.hidden, num_layers=a.layers, batch_first=True,
                            dropout=0.1 if a.layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2*a.hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, x):
        h = self.inp(x)
        h,_ = self.lstm(h)
        return self.head(h).squeeze(-1)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(Fdim, a.dmodel)
        self.pos = nn.Parameter(torch.randn(1,80,a.dmodel)*0.02)
        layer = nn.TransformerEncoderLayer(a.dmodel, a.heads, dim_feedforward=4*a.dmodel,
                                           dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, a.tlayer)
        self.norm = nn.LayerNorm(a.dmodel)
        self.head = nn.Sequential(nn.Linear(a.dmodel, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, x):
        h = self.inp(x) + self.pos
        h = self.enc(h)
        return self.head(self.norm(h)).squeeze(-1)

if a.model=='lstm':
    model=LSTMNet().to(dev)
elif a.model=='transformer':
    model=TransformerNet().to(dev)
else:
    model=ConformerNet().to(dev)
print('params', sum(q.numel() for q in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_epoch = int(np.ceil(len(tr_idx)/a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs*steps_per_epoch,
                                            pct_start=0.1, div_factor=10, final_div_factor=100)
mask = (uog==0).float()
scale = mask.sum() / mask.numel()

def masked_l1(pred, tgt, m):
    l = torch.abs(pred-tgt)*m
    return l.sum()/(m.sum()+1e-8)

def evaluate(idx):
    model.eval(); preds=[]; losses=[]
    with torch.no_grad():
        for i in range(0,len(idx),1024):
            b=torch.tensor(idx[i:i+1024],device=dev)
            pr=model(Xg[b])
            preds.append(pr.cpu())
            losses.append(masked_l1(pr,yg[b],mask[b])*len(b))
    preds=torch.cat(preds).numpy()
    mae_insp = float(sum(losses)/len(idx))
    mae_all = float(np.abs(preds - y[idx].numpy()).mean())
    return mae_insp, mae_all, preds

best=1e9; best_state=None; t0=time.time(); cand=[]
for ep in range(a.epochs):
    model.train()
    perm=np.random.permutation(tr_idx)
    tot=0.0; nb=0
    for i in range(0,len(perm),a.bs):
        b=torch.tensor(perm[i:i+a.bs],device=dev)
        pred=model(Xg[b])
        loss=masked_l1(pred,yg[b],mask[b])
        if a.exp_w>0:
            loss=loss+a.exp_w*torch.abs(pred-yg[b]).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    vmae, vall, _=evaluate(va_idx)
    if vmae<best:
        best=vmae; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        cand.append((vmae,best_state))
        cand=cand[-5:]
    if ep%5==0 or ep==a.epochs-1:
        print(f'ep {ep} loss {tot/nb:.4f} val_insp {vmae:.5f} val_all {vall:.5f} best {best:.5f} {time.time()-t0:.0f}s',flush=True)

model.load_state_dict(best_state)
b_insp,b_all,_=evaluate(va_idx)
print(f'BEST_SINGLE insp {b_insp:.5f} all {b_all:.5f}',flush=True)
if len(cand)>1:
    saved=model.state_dict()
    avg={k: torch.stack([c[1][k].float() for c in cand]).mean(0) for k in saved}
    model.load_state_dict(avg)
    a_insp,a_all,_=evaluate(va_idx)
    print(f'SWA{len(cand)} insp {a_insp:.5f} all {a_all:.5f}',flush=True)
    if a_insp<b_insp:
        best_state=avg
model.load_state_dict(best_state)
vmae, vall, vpred = evaluate(va_idx)
# test pred
model.eval(); tpreds=[]
with torch.no_grad():
    for i in range(0,len(Xteg),2048):
        tpreds.append(model(Xteg[i:i+2048]).cpu())
tpred=torch.cat(tpreds).numpy()
# reorder test preds from sorted-breath order to original/sample_submission block order
import pandas as pd
_orig=pd.read_csv('../../data/test.csv',usecols=['breath_id']).values.reshape(-1,80)[:,0]
_bid=dt['breath_id']; _pos={b:i for i,b in enumerate(_bid)}
tpred=tpred[[_pos[b] for b in _orig]]
name=f"{a.model}{a.tag}_f{a.fold}_s{a.seed}"
np.save(f'preds/oof_{name}.npy', vpred)
np.save(f'preds/valid_idx_{name}.npy', va_idx)
np.save(f'preds/test_{name}.npy', tpred)
torch.save(best_state, f'models/{name}.pt')
print(f'FINAL {name} val_insp {vmae:.5f} val_all {vall:.5f}',flush=True)
