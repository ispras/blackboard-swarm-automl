import os, sys, time, json, gc
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__))
FEAT2=os.path.join(HERE,'..','fomalhaut_omicron','cache')
CACHE=os.path.join(HERE,'cache')
os.makedirs(CACHE,exist_ok=True)

META={'contact_id','game_play','datetime','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','game_key'}
CAT=['is_ground','pos1_cat','pos2_cat']

DELTA_PREFIXES=['top_e1','height_e1','area_e1','cy_e1','cx_e1','bottom_e1',
                'top_s1','height_s1','area_s1','cy_s1','cx_s1','bottom_s1',
                'top_e2','height_e2','area_e2','cy_e2','cx_e2','bottom_e2',
                'top_s2','height_s2','area_s2','cy_s2','cx_s2','bottom_s2',
                'speed1','acceleration1','sa1','orientation1','direction1','speed2']

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P
    fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); lo=ps[i]; hi=ps[min(i+1,len(ps)-1)]
    return float(mcc[i]), float((lo+hi)/2)

ROLL_COLS=['top_e1','height_e1','area_e1','cy_e1','top_s1','height_s1','area_s1',
           'top_e2','height_e2','top_s2','height_s2','speed1','orientation1','speed2']

def add_deltas(d):
    d=d.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g=d.groupby(['game_play','p1','p2'],sort=False)
    new={}
    for c in DELTA_PREFIXES:
        if c not in d.columns: continue
        new[c+'_d1']=(d[c]-g[c].shift(1)).astype('float32')
        new[c+'_dp1']=(g[c].shift(-1)-d[c]).astype('float32')
    for c in ROLL_COLS:
        if c not in d.columns: continue
        new[c+'_rm3']=g[c].transform(lambda s:s.rolling(3,min_periods=1,center=True).mean()).astype('float32')
    X=pd.DataFrame(new,index=d.index)
    return pd.concat([d,X],axis=1)

def feats_of(d):
    return [c for c in d.columns if c not in META]

def main():
    t0=time.time()
    kind=sys.argv[1] if len(sys.argv)>1 else 'ground'   # ground | pair | all
    mode=sys.argv[2] if len(sys.argv)>2 else 'train'
    d=pd.read_parquet(os.path.join(FEAT2,f'feat2_{mode}.parquet'))
    d['is_ground']=(d.p2=='G').astype(np.int8)
    d['game_key']=d.game_play.str.split('_').str[0]
    d=add_deltas(d)
    if kind!='all':
        want=1 if kind=='ground' else 0
        d=d[d.is_ground==want].reset_index(drop=True)
    d.to_parquet(os.path.join(CACHE,f'featg_{kind}_{mode}.parquet'))
    print(mode,kind,'shape',d.shape,'time',round(time.time()-t0,1))

if __name__=='__main__':
    main()
