import numpy as np, pandas as pd, os, time, sys

DATA = '../../data'
OUT = sys.argv[1] if len(sys.argv)>1 else 'cache'
os.makedirs(OUT, exist_ok=True)

CONT = ['u_in','time_step','dt','u_in_cumsum','u_in_cummean','volume','area',
        'u_in_lag1','u_in_lag2','u_in_lag3','u_in_lag4',
        'u_in_diff1','u_in_diff2','u_in_diff3',
        'u_in_rollmean4','u_in_rollstd4','u_in_cummax','u_in_cummin',
        'time_frac','step','R','C','invR','invC','RC',
        'u_in_R','u_in_div_C','area_R','area_div_C','cumsum_R','cumsum_div_C',
        'u_in_lead1','u_in_lead2','u_in_lead3','u_out_cumsum','b_total_vol','b_u_in_mean','b_u_in_std','b_u_in_max','b_insp_len']
ONEHOT = ['R5','R20','R50','C10','C20','C50']

def build(df):
    df = df.sort_values(['breath_id','time_step']).reset_index(drop=True)
    g = df.groupby('breath_id', sort=False)
    df['dt'] = g['time_step'].diff().fillna(0.0)
    df['step'] = g.cumcount()
    df['time_frac'] = df['time_step'] / g['time_step'].transform('max')
    df['u_in_cumsum'] = g['u_in'].cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / (df['step']+1)
    df['area'] = (df['u_in']*df['dt']).groupby(df['breath_id'], sort=False).cumsum()
    df['volume'] = df['area']
    for L in (1,2,3,4):
        df[f'u_in_lag{L}'] = g['u_in'].shift(L).fillna(0.0)
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in_lag1'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in_lag2'] - df['u_in_lag3']
    lags4 = [df['u_in']] + [df[f'u_in_lag{L}'] for L in (1,2,3)]
    cnt = df['step'].clip(upper=3) + 1
    sm = sum(lags4)
    df['u_in_rollmean4'] = sm / cnt
    sq = sum(x*x for x in lags4)
    df['u_in_rollstd4'] = np.sqrt((sq/cnt - df['u_in_rollmean4']**2).clip(lower=0))
    df['u_in_cummax'] = g['u_in'].cummax()
    for L in (1,2,3):
        df[f'u_in_lead{L}'] = g['u_in'].shift(-L).fillna(0.0)
    df['u_out_cumsum'] = g['u_out'].cumsum()
    df['b_total_vol'] = g['area'].transform('max')
    df['b_u_in_mean'] = g['u_in'].transform('mean')
    df['b_u_in_std'] = g['u_in'].transform('std').fillna(0.0)
    df['b_u_in_max'] = g['u_in'].transform('max')
    df['b_insp_len'] = g['u_out'].transform(lambda s: (s==0).sum()).astype(np.float32)
    df['u_in_cummin'] = g['u_in'].cummin()
    df['invR'] = 1.0/df['R']; df['invC'] = 1.0/df['C']; df['RC'] = df['R']*df['C']
    df['u_in_R'] = df['u_in']*df['R']
    df['u_in_div_C'] = df['u_in']/df['C']
    df['area_R'] = df['area']*df['R']
    df['area_div_C'] = df['area']/df['C']
    df['cumsum_R'] = df['u_in_cumsum']*df['R']
    df['cumsum_div_C'] = df['u_in_cumsum']/df['C']
    for v in (5,20,50):
        df[f'R{v}'] = (df['R']==v).astype(np.float32)
    for v in (10,20,50):
        df[f'C{v}'] = (df['C']==v).astype(np.float32)
    return df

def main():
    t0=time.time()
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    tr = build(tr); te = build(te)
    cols = CONT + ONEHOT
    # z-score continuous using train stats
    mu = tr[CONT].mean(); sd = tr[CONT].std().replace(0,1)
    for c in CONT:
        tr[c] = (tr[c]-mu[c])/sd[c]
        te[c] = (te[c]-mu[c])/sd[c]
    ntr = tr.breath_id.nunique(); nte = te.breath_id.nunique()
    nsteps = 80
    assert len(tr)==ntr*nsteps and len(te)==nte*nsteps
    Xtr = tr[cols].to_numpy(np.float32).reshape(ntr,nsteps,len(cols))
    Xte = te[cols].to_numpy(np.float32).reshape(nte,nsteps,len(cols))
    ytr = tr['pressure'].to_numpy(np.float32).reshape(ntr,nsteps)
    utr = tr['u_out'].to_numpy(np.float32).reshape(ntr,nsteps)
    ute = te['u_out'].to_numpy(np.float32).reshape(nte,nsteps)
    btr = tr['breath_id'].to_numpy().reshape(ntr,nsteps)[:,0]
    bte = te['breath_id'].to_numpy().reshape(nte,nsteps)[:,0]
    np.savez_compressed(f'{OUT}/train.npz', X=Xtr, y=ytr, u_out=utr, breath_id=btr)
    np.savez_compressed(f'{OUT}/test.npz', X=Xte, u_out=ute, breath_id=bte)
    # folds: GroupKFold-like, seed 42 deterministic shuffle over sorted unique ids
    rng = np.random.RandomState(42)
    ids = np.sort(np.unique(btr))
    perm = rng.permutation(len(ids))
    fold_of = np.empty(len(ids), np.int8)
    fold_of[perm] = np.arange(len(ids)) % 5
    id2fold = dict(zip(ids, fold_of))
    folds = np.array([id2fold[b] for b in btr], np.int8)
    np.save(f'{OUT}/folds.npy', folds)
    json_cols = cols
    import json
    json.dump({'cols':cols,'mu':mu.to_dict(),'sd':sd.to_dict()}, open(f'{OUT}/meta.json','w'))
    print('Xtr',Xtr.shape,'Xte',Xte.shape,'folds',np.bincount(folds),'time',time.time()-t0)

if __name__=='__main__':
    main()
