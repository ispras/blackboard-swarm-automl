import argparse, os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from common import (load_train, load_test, load_train_extra, load_test_extra,
                    mcrmse, SCORE_IDX, ALL_TARGETS, SCORED_LEN, SEQ_LEN, WS)
from model import RNAModel

COMP = np.array([2, 3, 0, 1])  # A->U, G->C, U->A, C->G
ST_SWAP = np.array([0, 2, 1])


def revcomp_arrays(seq, struct, loop):
    return COMP[seq][:, ::-1].copy(), ST_SWAP[struct][:, ::-1].copy(), loop[:, ::-1].copy()


def predict_dl(model, dl, device):
    model.eval(); ps = []
    with torch.no_grad():
        for b in dl:
            xb = b[0] if isinstance(b, (list, tuple)) else b
            ps.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(ps)


def predict_tta(model, seq, struct, loop, extra, device, bs, mode=1):
    dl = DataLoader(DS(seq, struct, loop, extra=extra), batch_size=bs, shuffle=False, num_workers=2)
    p = predict_dl(model, dl, device)
    rs, rt, rl = revcomp_arrays(seq, struct, loop)
    dlr = DataLoader(DS(rs, rt, rl, extra=None), batch_size=bs, shuffle=False, num_workers=2)
    pr = predict_dl(model, dlr, device)
    if mode == 1:
        p_rc = pr[:, ::-1, :]
    else:
        # map only within scored window: original i <- pr[67-i] for i<68, else keep p
        p_rc = p.copy()
        p_rc[:, :68, :] = pr[:, 67::-1, :]
    return 0.5 * (p + p_rc)


class DS(Dataset):
    def __init__(self, seq, struct, loop, y=None, idx=None, extra=None, aug=False, ew=None):
        self.seq, self.struct, self.loop, self.y, self.idx, self.extra = seq, struct, loop, y, idx, extra
        self.aug = aug; self.ew = ew

    def __len__(self):
        return len(self.seq) if self.idx is None else len(self.idx)

    def __getitem__(self, i):
        j = i if self.idx is None else self.idx[i]
        parts = [np.eye(4, dtype=np.float32)[self.seq[j]],
                 np.eye(3, dtype=np.float32)[self.struct[j]],
                 np.eye(7, dtype=np.float32)[self.loop[j]]]
        if self.extra is not None:
            parts.append(self.extra[j])
        x = np.concatenate(parts, axis=1)
        if self.y is None:
            return torch.from_numpy(x)
        y = self.y[j]
        ewrow = None if self.ew is None else self.ew[i]
        if self.aug and self.extra is None and random.random() < 0.5:
            x = x[::-1].copy()
            if ewrow is not None:
                ewrow = ewrow[::-1].copy()
            if self.aug >= 2:
                seqc = x[:, :4].copy()
                x[:, 0] = seqc[:, 2]; x[:, 2] = seqc[:, 0]
                x[:, 1] = seqc[:, 3]; x[:, 3] = seqc[:, 1]
            c = x[:, 5].copy(); x[:, 5] = x[:, 6]; x[:, 6] = c
            y = y[::-1].copy()
        if ewrow is not None:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ewrow)
        return torch.from_numpy(x), torch.from_numpy(y)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.extra:
        tr, te = load_train_extra(), load_test_extra()
        in_dim = 14 + tr['extra'].shape[2]
    else:
        tr, te = load_train(), load_test()
        in_dim = 14
    n = len(tr['seq'])
    kf = KFold(n_splits=5, shuffle=True, random_state=args.split_seed)
    folds = list(kf.split(np.arange(n)))
    tr_idx, va_idx = folds[args.fold]
    snf = np.array([r['SN_filter'] for r in tr['rows']], dtype=bool)
    sn_all = tr['sn']
    if args.sn_only:
        tr_idx = tr_idx[snf[tr_idx]]

    # normalize targets per-target using fold-train stats
    y = tr['y'].copy()
    mask = tr['mask']
    flat = y[tr_idx][mask[tr_idx] > 0]
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    yn = (y - mu) / sd
    yn = yn * mask[..., None]  # zero-out unscored positions

    ex_tr = tr.get('extra')
    ew_all = None
    if args.errw:
        err_keys = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
                    'deg_error_Mg_50C', 'deg_error_50C']
        ew_all = np.zeros((n, SEQ_LEN, 5), dtype=np.float32)
        for i, r in enumerate(tr['rows']):
            for k, key in enumerate(err_keys):
                e = np.asarray(r[key], dtype=np.float32)
                c = {1: 1.0, 2: 0.25, 3: 4.0, 4: 1.0, 5: 1.0, 6: 1e-4, 7: 0.05, 8: 0.02}.get(args.errw, 1.0)
                wv = 1.0 / (e ** 2 + c)
                if args.errw == 4:
                    wv = np.minimum(wv, 5.0)
                elif args.errw == 5:
                    wv = np.minimum(wv, 10.0)
                elif args.errw == 6:
                    wv = np.minimum(wv, 5.0)
                elif args.errw == 7:
                    wv = np.minimum(wv, 5.0)
                ew_all[i, :len(e), k] = wv
        for k in range(5):
            v = ew_all[:, :SCORED_LEN, k]
            ew_all[:, :SCORED_LEN, k] = v / (v.mean() + 1e-9)
        if args.errw == 8:
            ew_all[:, :SCORED_LEN, :] = np.clip(ew_all[:, :SCORED_LEN, :], 0.05, 5.0)
    ds_tr = DS(tr['seq'], tr['struct'], tr['loop'], yn, tr_idx, ex_tr, aug=bool(args.aug),
               ew=(ew_all[tr_idx] if ew_all is not None else None))
    ds_va = DS(tr['seq'], tr['struct'], tr['loop'], yn, va_idx, ex_tr)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs * 2, shuffle=False, num_workers=2)

    model = RNAModel(args.arch, hidden=args.hidden, layers=args.layers,
                     drop=args.drop, in_dim=in_dim, bn=bool(args.bn)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.onecycle:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr), pct_start=0.1)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss(reduction='none')
    w = (torch.tensor([1., 1., 0.5, 1., 0.5], device=device) if args.aux
         else torch.tensor([1., 1., 0., 1., 0.], device=device))
    mw = torch.tensor(mask, device=device)

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    swa_sum = None; swa_n = 0
    best = 1e9
    best_state = None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in dl_tr:
            if len(batch) == 3:
                xb, yb, ewb = batch; ewb = ewb.to(device)
            else:
                xb, yb = batch; ewb = None
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            m = mw[:xb.shape[0]] if False else None
            # per-sample mask: DS preserves order within tr_idx; use global mask by idx missing here,
            # so recompute mask from yb nonzero padding is unsafe. We pass mask via yb size only.
            # Instead store masks in dataset target padding: all rows same 68 scored.
            bmask = torch.zeros_like(pred)
            bmask[:, :SCORED_LEN, :] = 1.0
            ww = bmask * w
            if ewb is not None:
                ww = ww * ewb
            if args.mcr:
                el = lossf(pred, yb)
                ms = []
                for c in SCORE_IDX:
                    wc = ww[:, :, c]
                    ms.append(torch.sqrt((el[:, :, c] * wc).sum() / (wc.sum() + 1e-9) + 1e-9))
                l = sum(ms) / len(ms)
            else:
                l = (lossf(pred, yb) * ww).sum() / (ww.sum() + 1e-9)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if args.ema > 0:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1 - args.ema)
        sched.step()
        if args.swa > 0 and ep >= args.epochs - args.swa:
            sd_ = model.state_dict()
            if swa_sum is None:
                swa_sum = {k: v.detach().float().clone() for k, v in sd_.items()}
            else:
                for k, v in sd_.items():
                    swa_sum[k] += v.detach().float()
            swa_n += 1

        # validation on raw scale
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb in dl_va:
                ps.append(model(xb.to(device)).cpu().numpy())
        p = np.concatenate(ps)
        p = p * sd + mu
        yv = tr['y'][va_idx]
        mv = mask[va_idx]
        sc, per = mcrmse(yv, p, mv)
        sfm = snf[va_idx]
        sc_sn, per_sn = (mcrmse(yv[sfm], p[sfm], mv[sfm]) if sfm.sum() > 5 else (float('nan'), []))
        if sc < best:
            best = sc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep} loss {l.item():.4f} val {sc:.4f} best {best:.4f} "
                  f"sn {sc_sn:.4f} ({time.time()-t0:.1f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    if args.ema > 0:
        ema_state = {k: v for k, v in ema.items()}
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb in dl_va:
                ps.append(model(xb.to(device)).cpu().numpy())
        pe = np.concatenate(ps) * sd + mu
        sce, _ = mcrmse(tr['y'][va_idx], pe, mask[va_idx])
        print(f"EMA val {sce:.4f} vs best {best:.4f}", flush=True)
        if sce < best:
            best = sce
            best_state = {k: v.detach().clone() for k, v in ema_state.items()}
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(cur)
    if args.swa > 0 and swa_n > 0:
        swa_state = {k: (v / swa_n) for k, v in swa_sum.items()}
        cur2 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(swa_state); model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb in dl_va:
                ps.append(model(xb.to(device)).cpu().numpy())
        psw = np.concatenate(ps) * sd + mu
        scw, _ = mcrmse(tr['y'][va_idx], psw, mask[va_idx])
        print(f"SWA val {scw:.4f} vs best {best:.4f} (n={swa_n})", flush=True)
        if scw < best:
            best = scw
            best_state = {k: v.detach().clone() for k, v in swa_state.items()}
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(cur2)
    # oof preds
    ps = []
    with torch.no_grad():
        for xb, yb in dl_va:
            ps.append(model(xb.to(device)).cpu().numpy())
    pv = np.concatenate(ps) * sd + mu
    if args.tta:
        pvt = predict_tta(model, tr['seq'][va_idx], tr['struct'][va_idx], tr['loop'][va_idx],
                          None, device, args.bs * 2, args.tta) * sd + mu
        sct, pert = mcrmse(tr['y'][va_idx], pvt, mask[va_idx])
        sfmt = snf[va_idx]
        sct_sn, pert_sn = mcrmse(tr['y'][va_idx][sfmt], pvt[sfmt], mask[va_idx][sfmt])
        print(f"TTA val {sct:.4f} per {np.round(pert,3)} | sn {sct_sn:.4f} per {np.round(pert_sn,3)}", flush=True)
        pv = pvt
    # test preds (average over 2 TTA: raw)
    dst = DS(te['seq'], te['struct'], te['loop'], extra=te.get('extra'))
    dlt = DataLoader(dst, batch_size=args.bs * 2, shuffle=False, num_workers=2)
    if args.tta:
        pt = predict_tta(model, te['seq'], te['struct'], te['loop'], te.get('extra'), device, args.bs * 2, args.tta) * sd + mu
    else:
        ps = []
        with torch.no_grad():
            for xb in dlt:
                ps.append(model(xb.to(device)).cpu().numpy())
        pt = np.concatenate(ps) * sd + mu

    os.makedirs(os.path.join(WS, 'preds'), exist_ok=True)
    tag = args.tag or f"{args.arch}_h{args.hidden}_l{args.layers}_s{args.seed}_f{args.fold}"
    np.savez(os.path.join(WS, 'preds', tag + '.npz'),
             val_idx=va_idx, oof=pv.astype(np.float32), test=pt.astype(np.float32),
             score=best)
    sfm = snf[va_idx]
    sc_sn, per_sn = mcrmse(tr['y'][va_idx][sfm], pv[sfm], mask[va_idx][sfm])
    print(f"DONE {tag} best {best:.4f} per {per} | sn {sc_sn:.4f} per {per_sn}", flush=True)
    with open(os.path.join(WS, 'preds', 'runs.log'), 'a') as f:
        f.write(json.dumps(dict(tag=tag, score=best, per=per, sn_score=sc_sn, sn_per=per_sn,
                                args=vars(args))) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='lstm')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--drop', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--sn_only', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--extra', type=int, default=1)
    ap.add_argument('--aux', type=int, default=1)
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--tta', type=int, default=0)
    ap.add_argument('--aug', type=int, default=0)
    ap.add_argument('--onecycle', type=int, default=0)
    ap.add_argument('--bn', type=int, default=1)
    ap.add_argument('--swa', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--errw', type=int, default=0)
    ap.add_argument('--mcr', type=int, default=0)
    run(ap.parse_args())
