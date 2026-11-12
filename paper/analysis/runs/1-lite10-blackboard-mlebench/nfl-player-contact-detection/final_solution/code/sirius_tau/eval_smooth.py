import pandas as pd, numpy as np, sys
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

FT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
FS='workspaces/fomalhaut_omicron/cache/feat_test.parquet'
cols=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact']
df=pd.read_parquet(FT,columns=cols)
oof=np.load('workspaces/sirius_tau/oof_hel.npy')
df['p']=oof
df['isg']=(df.nfl_player_id_2.astype(str)=='G').astype(int)
# sort so groups contiguous for rolling
df=df.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).reset_index(drop=True)
g=df.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)

def report(name,p,y):
    thr,mcc=best_thr(y,p); print(f'{name:28s} MCC {mcc:.5f} thr {thr:.4f}')

y=df.contact.values.astype(int)
report('raw', df.p.values, y)
for w in [3,5,7,9]:
    for how in ['mean','max']:
        s=g['p'].transform(lambda x: getattr(x.rolling(w,center=True,min_periods=1),how)())
        report(f'{how}{w}', s.values, y)
# per-type thresholds: ground and pair separate
for w in [5,7]:
    s=g['p'].transform(lambda x: x.rolling(w,center=True,min_periods=1).mean())
    for how2 in ['mean','max']:
        ss=g['p'].transform(lambda x: getattr(x.rolling(w,center=True,min_periods=1),how2)())
        # tune thr per type then combine
        thrg,_=best_thr(y[df.isg.values==1], ss.values[df.isg.values==1])
        thrp,_=best_thr(y[df.isg.values==0], ss.values[df.isg.values==0])
        pred=np.where(df.isg.values==1, ss.values>=thrg, ss.values>=thrp).astype(int)
        from sklearn.metrics import matthews_corrcoef
        print(f'per-type w{w} {how2}: MCC {matthews_corrcoef(y,pred):.5f} thrg {thrg:.4f} thrp {thrp:.4f}')
