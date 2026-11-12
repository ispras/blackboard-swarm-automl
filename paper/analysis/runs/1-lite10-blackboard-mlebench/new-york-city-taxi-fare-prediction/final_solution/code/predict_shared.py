import sys, os, argparse, json
import numpy as np, pandas as pd
ROOT='/download_job_code/payload/sandbox'
MIM=f'{ROOT}/workspaces/mimosa_omicron'; ME=f'{ROOT}/workspaces/aldebaran_omicron'
sys.path.insert(0,MIM)
from test_features import build_test, add_counts_te
from common import load_te, FEATS_V2, TE_COLS
ALLF=FEATS_V2+TE_COLS

def block(name, modelpath, out):
    te=load_te(); df=build_test(); df=add_counts_te(df,te)
    corrupt=df['corrupt'].to_numpy().astype(bool)
    if modelpath.endswith('.json') or modelpath.endswith('.ubj'):
        import xgboost as xgb
        b=xgb.Booster(); b.load_model(modelpath)
        names=b.feature_names if b.feature_names else ALLF
        X=df[names].to_numpy(np.float32)
        p=np.asarray(b.predict(xgb.DMatrix(X,feature_names=names)),dtype=np.float32)
    else:
        import catboost as cb
        X=df[ALLF].to_numpy(np.float32)
        m=cb.CatBoostRegressor(); m.load_model(modelpath)
        p=np.asarray(m.predict(X),dtype=np.float32)
    print(name,'valid mean',p[~corrupt].mean(),'corrupt',int(corrupt.sum()),flush=True)
    np.save(f'{ME}/test_pred_{name}.npy',p)
    np.save(f'{ME}/test_corrupt_mask.npy',corrupt)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',default=None)
    a=ap.parse_args(); block(a.name,a.model,a.out)
