import pandas as pd, numpy as np, cv2, os
FPS=59.94
md=pd.read_csv('data/train_video_metadata.csv')
META={}
for _,r in md.iterrows():
    st=pd.Timestamp(r.start_time); sn=pd.Timestamp(r.snap_time)
    META[(r.game_play,r.view)]=(sn-st).total_seconds()*FPS
h=pd.read_parquet('workspaces/fomalhaut_omicron/cache/hel_train.parquet')
print('hel cols',h.columns.tolist())
print(h[h.game_play=='58168_003392'].head())
gp='58168_003392'
for v in ['Endzone','Sideline']:
    print(v,'base',META.get((gp,v)))
sub=h[(h.game_play==gp)&(h.view=='Endzone')]
print('steps',sorted(sub.step.unique())[:10], '...', sorted(sub.step.unique())[-5:])
print('players', sorted(sub.nfl_player_id.unique())[:10])
idx=pd.read_parquet('workspaces/sirius_tau/crops_index.parquet')
s=idx[idx.game_play==gp].iloc[0]
print('sample', s.to_dict())
step=int(s.step)
cap=cv2.VideoCapture(f'data/train/{gp}_Endzone.mp4')
print('opened',cap.isOpened())
# read frame for step
base=META[(gp,'Endzone')]
fr=int(round(base+step*0.1*FPS)); print('frame',fr)
cap.set(cv2.CAP_PROP_POS_FRAMES,fr)
ok,img=cap.read(); print('read ok',ok, None if not ok else img.shape, None if not ok else img.mean())
cap.release()
