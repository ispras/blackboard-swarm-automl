import pandas as pd, numpy as np, sys
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

FT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
FS='workspaces/fomalhaut_omicron/cache/feat_test.parquet'
W=7

df=pd.read_parquet(FT,columns=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact'])
df['p']=np.load('workspaces/sirius_tau/oof_hel.npy')
df=df.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).reset_index(drop=True)
g=df.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)
sm=g['p'].transform(lambda x: x.rolling(W,center=True,min_periods=1).mean())
thr,mcc=best_thr(df.contact.values.astype(int), sm.values)
print('smoothed OOF', mcc, 'thr', thr)

te=pd.read_parquet(FS,columns=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2'])
te['p']=np.load('workspaces/sirius_tau/test_hel.npy')
te=te.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).reset_index(drop=True)
gt=te.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)
te['ps']=gt['p'].transform(lambda x: x.rolling(W,center=True,min_periods=1).mean())
sub=pd.read_csv('data/sample_submission.csv').merge(te[['contact_id','ps']],on='contact_id',how='left')
sub['contact']=(sub.ps>=thr).astype(int)
print('pos rate', sub.contact.mean(), 'nan', sub.ps.isna().sum())
sub[['contact_id','contact']].to_csv('workspaces/sirius_tau/sub_hel_sm.csv',index=False)
print('saved')
