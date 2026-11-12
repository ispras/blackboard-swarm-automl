import sys, os, json, glob
import numpy as np
import pandas as pd

ROOT='/download_job_code/payload/sandbox'
ME=f'{ROOT}/workspaces/aldebaran_omicron'
MIM=f'{ROOT}/workspaces/mimosa_omicron'
CAN=f'{ROOT}/workspaces/canopus_pi'
NTEST=9914

def find_test(src,tag):
    if src=='alde':
        for f in [f'{ME}/test_pred_{tag}.npy', f'{ME}/submission_{tag}.csv']:
            if os.path.exists(f):
                return np.load(f) if f.endswith('.npy') else pd.read_csv(f)['fare_amount'].to_numpy()
    if src=='mimosa':
        for f in [f'{ME}/test_pred_mimosa_{tag}.npy', f'{MIM}/test_pred_{tag}.npy', f'{MIM}/submission_{tag}.csv']:
            if os.path.exists(f):
                return np.load(f) if f.endswith('.npy') else pd.read_csv(f)['fare_amount'].to_numpy()
    if src=='canopus':
        for f in [f'{ME}/test_pred_canopus_{tag}.npy', f'{CAN}/preds/test_pred_{tag}.npy', f'{CAN}/models/test_pred_{tag}.npy', f'{CAN}/test_pred_{tag}.npy', f'{CAN}/preds/submission_{tag}.csv', f'{CAN}/submission_{tag}.csv']:
            if os.path.exists(f):
                return np.load(f) if f.endswith('.npy') else pd.read_csv(f)['fare_amount'].to_numpy()
    return None

def load_models():
    M={}
    for p in glob.glob(f'{ME}/holdout_pred_*.npy'):
        tag=os.path.basename(p)[len('holdout_pred_'):-4]; o=f'{ME}/holdout_orig_{tag}.npy'
        if os.path.exists(o): M[f'alde_{tag}']=dict(orig=np.load(o),pred=np.load(p),test=find_test('alde',tag))
    for p in glob.glob(f'{MIM}/holdout_pred_*.npy'):
        tag=os.path.basename(p)[len('holdout_pred_'):-4]; o=f'{MIM}/holdout_orig_{tag}.npy'
        if os.path.exists(o): M[f'mimosa_{tag}']=dict(orig=np.load(o),pred=np.load(p),test=find_test('mimosa',tag))
    for f in glob.glob(f'{CAN}/models/*_holdout.npz'):
        d=np.load(f); tag=os.path.basename(f)[:-len('_holdout.npz')]
        M[f'canopus_{tag}']=dict(orig=d['orig_idx'],pred=d['pred'],test=find_test('canopus',tag))
    for f in glob.glob(f'{CAN}/preds/holdout_pred_canopus_*.npy'):
        tag=os.path.basename(f)[len('holdout_pred_canopus_'):-4]
        o=f'{CAN}/preds/holdout_orig_canopus_{tag}.npy'
        if os.path.exists(o):
            M[f'canopus_{tag}']=dict(orig=np.load(o),pred=np.load(f),test=find_test('canopus',tag))
    return M

def main():
    M=load_models()
    for n,d in M.items():
        t=d['test']
        print(f'{n:30s} hold {len(d["orig"])} test {"None" if t is None else len(t)}')
    # only models with test preds
    names=[n for n,d in M.items() if d['test'] is not None and len(d['test'])==NTEST]
    if not names:
        print('no models with test preds'); return
    common=None
    for n in names:
        s=set(M[n]['orig'].tolist()); common=s if common is None else common & s
    common=np.array(sorted(common),dtype=np.int64)
    oref=np.load(f'{ME}/holdout_origidx.npy'); yref=np.load(f'{ME}/holdout_y.npy')
    ymap=dict(zip(oref.tolist(),yref.tolist()))
    res=[]
    for i in common:
        if int(i) in ymap: res.append(int(i))
    common=np.array(res,dtype=np.int64); y=np.array([ymap[i] for i in common])
    cols=[]
    for n in names:
        d=M[n]; s=pd.Series(np.arange(len(d['orig'])),index=d['orig'])
        pos=s.loc[common].to_numpy(); cols.append(d['pred'][pos].astype(np.float64))
    Pm=np.vstack(cols).T
    print('blend matrix',Pm.shape,'models',names)
    for j,n in enumerate(names):
        print(f'{n:30s} holdout rmse {np.sqrt(np.mean((Pm[:,j]-y)**2)):.5f}')
    from sklearn.model_selection import KFold
    from scipy.optimize import nnls
    def fit(Xt,yt,mode):
        if mode=='ols':
            A=np.column_stack([Xt,np.ones(len(Xt))]); w=np.linalg.lstsq(A,yt,rcond=None)[0]; return w
        w,_=nnls(Xt,yt); return w
    def apply(X,w,mode):
        if mode=='ols': return np.column_stack([X,np.ones(len(X))])@w
        return X@w
    kf=KFold(5,shuffle=True,random_state=0); best=None
    for mode in ['nnls','ols']:
        oof=np.zeros(len(y))
        for tr,te in kf.split(Pm): oof[te]=apply(Pm[te],fit(Pm[tr],y[tr],mode),mode)
        r=np.sqrt(np.mean((oof-y)**2)); print(mode,'CV',r)
        if best is None or r<best[1]-0.005: best=(mode,r)
    mode=best[0]
    if mode=='nnls':
        # bagged weights: average NNLS weights over 5 folds (more robust than single fit)
        ws=[]
        for tr,te in KFold(5,shuffle=True,random_state=0).split(Pm):
            ws.append(fit(Pm[tr],y[tr],mode))
        w=np.mean(ws,axis=0)
    else:
        w=fit(Pm,y,mode)
    print('chosen',mode,'weights',dict(zip(names,np.round(w,4))))
    sys.path.insert(0,f'{MIM}')
    from test_features import build_test, add_counts_te
    from common import load_te
    df=build_test(); df=add_counts_te(df,load_te()); corrupt=df['corrupt'].to_numpy().astype(bool)
    valid=~corrupt
    T=np.vstack([M[n]['test'] for n in names]).T
    pv=np.zeros(NTEST)
    if mode=='ols':
        ww=w[:-1]; b=w[-1]
        pv[valid]=T[valid]@ww + b
    else:
        pv[valid]=T[valid]@w
    # corrupt: use corrupt-trained model
    cp=None
    if os.path.exists(f'{ME}/corrupt_test_pred.npy'):
        cp=np.load(f'{ME}/corrupt_test_pred.npy')
    # prefer canopus corrupt model (better CV 12.67 vs 12.81) where it has a prediction
    canpos=f'{CAN}/preds/corrupt_test_positions.npy'; canpred=f'{CAN}/preds/corrupt_test_pred_canopus.npy'
    if cp is not None and os.path.exists(canpos) and os.path.exists(canpred):
        P=np.load(canpos).astype(int); V=np.load(canpred)
        carr=cp.copy()
        for p_,v_ in zip(P,V):
            if 0<=p_<NTEST and corrupt[p_]: carr[p_]=v_
        cp=carr
    cconst=12.2
    pv[corrupt]=cp[corrupt] if cp is not None else cconst
    pv=np.clip(pv,2.5,500.0)
    out=pd.DataFrame({'key':pd.read_csv(f'{ROOT}/data/test.csv')['key'],'fare_amount':pv})
    out.to_csv(f'{ME}/submission_blend.csv',index=False)
    np.save(f'{ME}/test_pred_blend.npy',pv)
    with open(f'{ME}/blend_result.json','w') as f:
        json.dump({'mode':mode,'cv_rmse':best[1],'weights':dict(zip(names,w.tolist())),
                   'corrupt_source':'corrupt_test_pred.npy' if cp is not None else cconst,'corrupt_n':int(corrupt.sum())},f,indent=2)
    print('saved blend valid mean',pv[valid].mean(),'corrupt',pv[corrupt].mean())

if __name__=='__main__':
    main()
