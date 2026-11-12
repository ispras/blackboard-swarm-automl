import pandas as pd, numpy as np, json, os, sys, time
sys.path.insert(0, '.')
from postprocess_search import _bounds, _roll_mean_center

R='.'; A='../achernar_sigma'; C='../castor_alpha'
GKEYS=['game_play','nfl_player_id_1','nfl_player_id_2']
cfg=json.load(open('ground_best.json'))
wd=cfg['wd']; names=cfg['names']; w=np.array(cfg['weights'],dtype=np.float64)

paths={
 'track':('testpred_ground.parquet','prob'),
 'ach2':(f'{A}/testpred_ground_lgbv2.parquet','pred'),
 'gfus':(f'{A}/testpred_ground_gfus.parquet','pred'),
 'ctab':(f'{C}/ground_test_tab.parquet','pred'),
 'cvid':(f'{C}/ground_test_video.parquet','pred'),
 'ceff':(f'{C}/ground_test_video_eff.parquet','pred'),
 'eb3':(f'{C}/ground_test_video_eb3.parquet','pred'),
 'cnx2':(f'{C}/ground_test_video_cnx2.parquet','pred'),
 'wide':(f'{C}/ground_test_video_wide.parquet','pred'),
 'tight':(f'{C}/ground_test_video_tight.parquet','pred'),
 'r34':(f'{C}/ground_test_video_r34.parquet','pred'),
 'convnext':(f'{C}/ground_test_video_cnx.parquet','pred'),
 'stackr':('ground_test_stack_r.parquet','prob'),
 'gstack':(f'{A}/testpred_ground_gstack.parquet','pred'),
 'gstack2':(f'{A}/testpred_ground_gstack2.parquet','pred'),
}
STALE_AFTER=time.time()-6*3600  # drop stale gstack2 if older than 30min
te=pd.read_parquet('testpred_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
keep_names=[]; keep_w=[]
for n,ww in zip(names,w):
    if n not in paths:
        continue
    p,c=paths[n]
    if not os.path.exists(p):
        print('skip missing',n); continue
    if n=='gstack2' and os.path.getmtime(p)<STALE_AFTER:
        print('skip stale gstack2', time.ctime(os.path.getmtime(p))); continue
    d=pd.read_parquet(p)[['contact_id',c]].rename(columns={c:n})
    if d.contact_id.duplicated().any():
        print('dedup',n); d=d.drop_duplicates('contact_id')
    te=te.merge(d,on='contact_id',how='left')
    keep_names.append(n); keep_w.append(ww)
    print(n,'mtime',time.ctime(os.path.getmtime(p)),'nan',int(te[n].isna().sum()))
keep_w=np.array(keep_w); keep_w=keep_w/keep_w.sum()
print('weights',dict(zip(keep_names,np.round(keep_w,4))))
p_test=sum(wj*te[n].values for n,wj in zip(keep_names,keep_w))
te['_p']=p_test
te=te.sort_values(GKEYS+['step'],kind='mergesort').reset_index(drop=True)
gid=te.groupby(GKEYS,sort=False).ngroup().values.astype(np.int64)
gs,ge=_bounds(gid)
sm=_roll_mean_center(te['_p'].values.astype(np.float32),gs,ge,wd)
pd.DataFrame({'contact_id':te.contact_id,'game_play':te.game_play,'nfl_player_id_1':te.nfl_player_id_1,
              'nfl_player_id_2':te.nfl_player_id_2,'step':te.step,'prob':sm}).to_parquet('ground_test_best.parquet',index=False)
print('saved ground_test_best.parquet rows',len(te),'posrate@.045',float((sm>=np.quantile(sm,0.955)).mean()))
