import os, sys, time, json, argparse
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
T = 80

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--nfold', type=int, default=5)
    p.add_argument('--epochs', type=int, default=130)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--head_layers', type=int, default=2)
    p.add_argument('--head_hidden', type=int, default=256)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=3e-4)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--exp_weight', type=float, default=0.1)
    p.add_argument('--tag', type=str, default='lstm3')
    p.add_argument('--all_data', action='store_true')
    p.add_argument('--save_last', action='store_true')
    p.add_argument('--xpre', type=str, default='X2')
    p.add_argument('--init', type=str, default='')
    p.add_argument('--pct_start', type=float, default=0.12)
    return p.parse_args()

class Net(nn.Module):
    def __init__(self, nf, hidden, layers, dropout=0.1, head_layers=2, head_hidden=256):
        super().__init__()
        self.lstm1 = nn.LSTM(nf, hidden, layers, batch_first=True, bidirectional=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.lstm2 = nn.LSTM(2*hidden, head_hidden, head_layers, batch_first=True, bidirectional=True,
                             dropout=dropout if head_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2*head_hidden, 256), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))
    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        return self.head(h).squeeze(-1)

def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'
    X = np.load(os.path.join(HERE, f'{a.xpre}_train.npy'))
    Xte = np.load(os.path.join(HERE, f'{a.xpre}_test.npy'))
    y = np.load(os.path.join(HERE,'y_train.npy'))
    m = np.load(os.path.join(HERE,'mask_train.npy'))
    bid = np.load(os.path.join(HERE,'breath_train.npy'))
    n = len(bid); nf = X.shape[1]
    X = X.reshape(n, T, nf); y = y.reshape(n, T); m = m.reshape(n, T)
    ymu = float(y[m > 0].mean()); ysd = float(y[m > 0].std())
    y = (y - ymu)/ysd
    w = m*(1.0 - a.exp_weight) + a.exp_weight  # insp 1.0, exp exp_weight
    fv = (bid % a.nfold).astype(np.int64)
    if a.all_data:
        tr_idx = np.arange(n); val_idx = np.where(fv == a.fold)[0]
    else:
        tr_idx = np.where(fv != a.fold)[0]; val_idx = np.where(fv == a.fold)[0]
    print(f'train {len(tr_idx)} val {len(val_idx)} nf {nf} y {ymu:.3f}/{ysd:.3f}', flush=True)

    Xg = torch.from_numpy(X[tr_idx]).to(dev); yg = torch.from_numpy(y[tr_idx]).to(dev); wg = torch.from_numpy(w[tr_idx]).to(dev)
    Xv = torch.from_numpy(X[val_idx]).to(dev)
    Xte_g = torch.from_numpy(Xte.reshape(-1, T, nf)).to(dev)
    yv_true = y[val_idx]*ysd + ymu; uv = m[val_idx]  # uv==1 marks inspiratory rows

    model = Net(nf, a.hidden, a.layers, a.dropout, a.head_layers, a.head_hidden).to(dev)
    if a.init and os.path.exists(a.init):
        model.load_state_dict(torch.load(a.init)); print('init from', a.init, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    ntr = len(tr_idx); steps = (ntr + a.batch - 1)//a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs*steps,
                                                pct_start=a.pct_start, div_factor=10.0, final_div_factor=100.0)
    scaler = torch.cuda.amp.GradScaler()
    best = 1e9; best_ep = -1; bad = 0
    ckpt = os.path.join(HERE, f'best_{a.tag}_f{a.fold}.pt')

    def predict(Xt, bs=512):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                with torch.cuda.amp.autocast():
                    out.append(model(Xt[i:i+bs]).float())
        return torch.cat(out).cpu().numpy()

    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(ntr, device=dev)
        tl = 0.0; nb = 0; t0 = time.time()
        for i in range(0, ntr, a.batch):
            sel = perm[i:i+a.batch]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(Xg[sel])
                loss = ((out.float() - yg[sel]).abs() * wg[sel]).sum() / (wg[sel].sum()+1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        vp = predict(Xv)*ysd + ymu
        vmae = float(np.abs(vp[uv == 1] - yv_true[uv == 1]).mean())
        print(f'ep {ep+1:3d} loss {tl/nb:.4f} val_mae {vmae:.5f} lr {opt.param_groups[0]["lr"]:.2e} {time.time()-t0:.1f}s', flush=True)
        if a.save_last:
            torch.save(model.state_dict(), ckpt)
            if vmae < best: best = vmae; best_ep = ep+1
        elif vmae < best:
            best = vmae; best_ep = ep+1; bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if bad >= a.patience:
            print('early stop', flush=True); break

    print(f'BEST {a.tag} fold{a.fold} val_mae {best:.5f} ep {best_ep}', flush=True)
    model.load_state_dict(torch.load(ckpt)); model.eval()
    vp = predict(Xv)*ysd + ymu
    tp = predict(Xte_g)*ysd + ymu
    vmae = float(np.abs(vp[uv == 1] - yv_true[uv == 1]).mean())
    print(f'FINAL {a.tag} fold{a.fold} val_mae {vmae:.5f}', flush=True)
    oof = np.full((n, T), np.nan, dtype=np.float32); oof[val_idx] = vp
    np.save(os.path.join(HERE, f'oof_{a.tag}_f{a.fold}.npy'), oof.reshape(-1))
    np.save(os.path.join(HERE, f'test_{a.tag}_f{a.fold}.npy'), tp.reshape(-1).astype(np.float32))
    np.save(os.path.join(HERE, f'valid_pred_{a.tag}_f{a.fold}.npy'), vp.reshape(-1).astype(np.float32))
    json.dump({'tag':a.tag,'fold':a.fold,'val_mae':vmae,'best_ep':best_ep,'all_data':a.all_data,
               'hidden':a.hidden,'layers':a.layers,'seed':a.seed,'epochs':a.epochs},
              open(os.path.join(HERE, f'result_{a.tag}_f{a.fold}.json'),'w'))

if __name__ == '__main__':
    main()
