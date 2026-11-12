import os, glob
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(os.path.dirname(__file__), 'members')
os.makedirs(OUT, exist_ok=True)
P = [f'p{i}' for i in range(5)]


def norm(d):
    s = d[P].sum(1).values.reshape(-1, 1)
    d[P] = d[P].values / np.clip(s, 1e-9, None)
    return d


def main():
    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]
    _, can_va = train_test_split(np.arange(len(df)), test_size=0.15,
                                 random_state=42, stratify=df.label.values)
    np.save(os.path.join(OUT, 'canonical_val_idx.npy'), can_va)

    # ---- altair effv2m 5-fold ----
    a = os.path.join(ROOT, 'workspaces', 'altair_chi', 'out')
    oof = np.zeros((len(df), 5), np.float32)
    got = np.zeros(len(df), bool)
    tests = []
    used = []
    for f in range(5):
        op = os.path.join(a, f'oof_effm_f{f}.npy')
        ip = os.path.join(a, f'oof_idx_effm_f{f}.npy')
        tp = os.path.join(a, f'test_effm_f{f}.npy')
        if os.path.exists(op) and os.path.exists(ip) and os.path.exists(tp):
            idx = np.load(ip)
            oof[idx] = np.load(op)
            got[idx] = True
            tests.append(np.load(tp))
            used.append(f)
    if used:
        o = pd.DataFrame(oof, columns=P); o.insert(0, 'image_id', df.image_id.values)
        o['label'] = df.label.values
        o = o[got]
        o.to_csv(os.path.join(OUT, 'altair_effv2m_oof.csv'), index=False)
        t = pd.DataFrame(np.mean(tests, 0), columns=P)
        t.insert(0, 'image_id', test.image_id.values)
        t.to_csv(os.path.join(OUT, 'altair_effv2m_test.csv'), index=False)
        acc = (oof[got].argmax(1) == df.label.values[got]).mean()
        print(f'altair effv2m folds {used} coverage {got.mean():.3f} OOF acc {acc:.4f}')
    else:
        print('altair effv2m: no folds ready')

    # ---- bellatrix swin (canonical val + test) ----
    b = os.path.join(ROOT, 'workspaces', 'bellatrix_tau', 'runs', 'swinb_384')
    vp = os.path.join(b, 'val_probs_rot8.csv')
    tp = os.path.join(b, 'test_probs_rot8.csv')
    if os.path.exists(vp) and os.path.exists(tp):
        v = pd.read_csv(vp)
        if 'label' not in v.columns:
            v['label'] = df.iloc[can_va].label.values if len(v) == len(can_va) else np.nan
        norm(v).to_csv(os.path.join(OUT, 'bellatrix_swin_oof.csv'), index=False)
        norm(pd.read_csv(tp)).to_csv(os.path.join(OUT, 'bellatrix_swin_test.csv'), index=False)
        print('bellatrix swin ->', len(v), 'val rows')

    # ---- bellatrix swin seed123 ----
    for tag, sub in [('swin_s123', 'swinb_384_s123')]:
        d = os.path.join(ROOT, 'workspaces', 'bellatrix_tau', 'runs', sub)
        for pref in ['rot8', 'hv']:
            vp = os.path.join(d, f'val_probs_{pref}.csv')
            tp = os.path.join(d, f'test_probs_{pref}.csv')
            if os.path.exists(vp) and os.path.exists(tp):
                v = pd.read_csv(vp)
                if 'label' not in v.columns:
                    v['label'] = df.iloc[can_va].label.values if len(v) == len(can_va) else np.nan
                norm(v).to_csv(os.path.join(OUT, f'bellatrix_{tag}_oof.csv'), index=False)
                norm(pd.read_csv(tp)).to_csv(os.path.join(OUT, f'bellatrix_{tag}_test.csv'), index=False)
                print(f'bellatrix {tag} ({pref}) ->', len(v), 'val rows')
                break

    # ---- bellatrix eva02 ----
    valid_test_ids = set(test.image_id)
    for tag, sub, pref in [('eva02', 'eva02_448', 'hv')]:
        d = os.path.join(ROOT, 'workspaces', 'bellatrix_tau', 'runs', sub)
        vp = os.path.join(d, f'val_probs_{pref}.csv')
        if not os.path.exists(vp):
            continue
        v = pd.read_csv(vp)
        if 'label' not in v.columns:
            v['label'] = df.iloc[can_va].label.values if len(v) == len(can_va) else np.nan
        norm(v).to_csv(os.path.join(OUT, f'bellatrix_{tag}_oof.csv'), index=False)
        # find a test csv: 2676 rows with sample_submission ids
        tp = None
        for cand in [f'test_probs_{pref}.csv', 'probs.csv', 'test_probs.csv',
                     'test_probs_rot8.csv', 'probs_test.csv']:
            p = os.path.join(d, cand)
            if os.path.exists(p):
                t = pd.read_csv(p)
                if len(t) == len(test) and set(t.image_id) == valid_test_ids:
                    tp = p
                    break
        if tp:
            norm(pd.read_csv(tp)).to_csv(os.path.join(OUT, f'bellatrix_{tag}_test.csv'), index=False)
            print(f'bellatrix {tag} -> val {len(v)} test from {os.path.basename(tp)}')
        else:
            print(f'bellatrix {tag}: val ready, NO TEST PROBS YET')

    # ---- bellatrix effnet b0 (canonical val from npy) ----
    b0 = os.path.join(ROOT, 'workspaces', 'bellatrix_tau', 'runs', 'b0_224')
    vnp, tcs = os.path.join(b0, 'val_probs.npy'), os.path.join(b0, 'probs.csv')
    if os.path.exists(vnp) and os.path.exists(tcs):
        vp = np.load(vnp)
        if len(vp) == len(can_va):
            v = pd.DataFrame(vp, columns=P)
            v.insert(0, 'image_id', df.iloc[can_va].image_id.values)
            v['label'] = df.iloc[can_va].label.values
            norm(v).to_csv(os.path.join(OUT, 'bellatrix_b0_oof.csv'), index=False)
            norm(pd.read_csv(tcs)).to_csv(os.path.join(OUT, 'bellatrix_b0_test.csv'), index=False)
            print('bellatrix b0 ->', len(v))

    # ---- altair single-split members (canonical 15%) ----
    for tag, vname, tname in [('altair_effml', 'effml_val_probs.csv', 'effml_test_probs.csv'),
                              ('altair_cnb', 'cnb_val_probs.csv', 'cnb_test_probs.csv')]:
        vp = os.path.join(ROOT, 'workspaces', 'altair_chi', vname)
        tp = os.path.join(ROOT, 'workspaces', 'altair_chi', tname)
        if os.path.exists(vp) and os.path.exists(tp):
            v = pd.read_csv(vp)
            if 'label' not in v.columns and len(v) == len(can_va):
                v = v.set_index('image_id').reindex(df.iloc[can_va].image_id.values).reset_index()
                v.columns = ['image_id'] + P
                v['label'] = df.iloc[can_va].label.values
            norm(v).to_csv(os.path.join(OUT, f'{tag}_oof.csv'), index=False)
            norm(pd.read_csv(tp)).to_csv(os.path.join(OUT, f'{tag}_test.csv'), index=False)
            print(f'{tag} ->', len(v), 'val rows')

    # ---- capella convnext members ----
    for tag, run in [('capella_cnxt_384', 'convnext_tiny_384'),
                     ('capella_cnxt_512ft', 'convnext_tiny_512_ft'),
                     ('capella_cnxt_small', 'convnext_small_384'),
                     ('capella_effb4', 'effnet_b4_384')]:
        d = os.path.join(ROOT, 'workspaces', 'capella_xi', 'runs', run)
        # prefer assembled full oof/test
        fo = os.path.join(d, 'oof_probs.csv')
        ft = os.path.join(d, 'test_probs.csv')
        if os.path.exists(fo) and os.path.exists(ft):
            norm(pd.read_csv(fo)).to_csv(os.path.join(OUT, f'{tag}_oof.csv'), index=False)
            norm(pd.read_csv(ft)).to_csv(os.path.join(OUT, f'{tag}_test.csv'), index=False)
            print(tag, 'assembled')
            continue
        # else per-fold npy
        oof = None
        tests = []
        used = []
        for f in range(5):
            vpf = os.path.join(d, f'val_probs_fold{f}.csv')
            tpf = os.path.join(d, f'test_probs_fold{f}.npy')
            if os.path.exists(vpf):
                v = pd.read_csv(vpf)
                if oof is None:
                    oof = pd.DataFrame({'image_id': df.image_id.values})
                    for c in P:
                        oof[c] = np.nan
                m = oof.image_id.isin(v.image_id)
                # assign by image_id order
                tmp = v.set_index('image_id').reindex(oof.image_id)
                for c in P:
                    oof[c] = np.where(oof[c].isna(), tmp[c].values, oof[c])
                if os.path.exists(tpf):
                    tests.append(np.load(tpf)); used.append(f)
        if oof is not None:
            oof['label'] = df.label.values
            oof = oof.dropna(subset=P)
            norm(oof).to_csv(os.path.join(OUT, f'{tag}_oof.csv'), index=False)
            if tests:
                t = pd.DataFrame(np.mean(tests, 0), columns=P)
                t.insert(0, 'image_id', test.image_id.values)
                norm(t).to_csv(os.path.join(OUT, f'{tag}_test.csv'), index=False)
            print(tag, 'fold(s)', used, 'coverage', len(oof) / len(df))


if __name__ == '__main__':
    main()
