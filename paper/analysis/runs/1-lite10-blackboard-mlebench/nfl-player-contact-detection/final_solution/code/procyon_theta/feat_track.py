import os, sys, time, json
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(OUT, exist_ok=True)

TRACK_COLS = ['game_play','game_key','play_id','nfl_player_id','step',
              'x_position','y_position','speed','distance','direction',
              'orientation','acceleration','sa','position','team']

def load_tracking(split):
    df = pd.read_csv(os.path.join(DATA, f'{split}_player_tracking.csv'), usecols=TRACK_COLS)
    return df

def player_derived(df):
    rad = np.deg2rad(df['direction'].values)
    df = df.copy()
    df['vx'] = df['speed'].values * np.sin(rad)
    df['vy'] = df['speed'].values * np.cos(rad)
    orad = np.deg2rad(df['orientation'].values)
    df['ox'] = np.sin(orad)
    df['oy'] = np.cos(orad)
    df['same_team'] = 0
    return df

def build_labels(split):
    if split == 'train':
        lab = pd.read_csv(os.path.join(DATA,'train_labels.csv'),
                          usecols=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact'])
    else:
        ss = pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
        parts = ss.contact_id.str.split('_')
        lab = pd.DataFrame({
            'game_play': parts.str[0] + '_' + parts.str[1],
            'step': parts.str[2].astype(int),
            'nfl_player_id_1': parts.str[3],
            'nfl_player_id_2': parts.str[4],
            'contact': 0,
        })
    return lab

def circ_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)

def team_scalar(s):
    return s.map({'home':0,'away':1}).astype('float32')

def merge_side(lab, tr, side):
    p = lab.copy()
    if side == 1:
        pid = lab['nfl_player_id_1'].astype(np.int64)
    else:
        pid = pd.to_numeric(lab['nfl_player_id_2'], errors='coerce')  # NaN for G
    p['nfl_player_id'] = pid
    cols = ['x_position','y_position','speed','distance','direction',
            'orientation','acceleration','sa','vx','vy','ox','oy']
    tr2 = tr[['game_play','step','nfl_player_id'] + cols]
    m = p.merge(tr2, on=['game_play','step','nfl_player_id'], how='left')
    ren = {c: f'{c}_{side}' for c in cols}
    m = m.rename(columns=ren)
    m = m.drop(columns=['nfl_player_id'])
    return m

def pair_features(df):
    out = pd.DataFrame(index=df.index)
    for c in ['x_position','y_position','speed','distance','acceleration','sa','vx','vy','ox','oy']:
        out[c+'_1'] = df[c+'_1'].astype('float32')
        out[c+'_2'] = df[c+'_2'].astype('float32')
    dx = df['x_position_1'] - df['x_position_2']
    dy = df['y_position_1'] - df['y_position_2']
    dist = np.sqrt(dx*dx + dy*dy)
    out['dx']=dx.astype('float32'); out['dy']=dy.astype('float32')
    out['dist']=dist.astype('float32')
    dvx = df['vx_1']-df['vx_2']; dvy=df['vy_1']-df['vy_2']
    # closing speed: rate of change of distance (negative = closing)
    denom = np.where(dist<1e-3, 1e-3, dist)
    out['closing_speed']=((dx*dvx+dy*dvy)/denom).astype('float32')
    out['rel_speed']=np.sqrt(dvx*dvx+dvy*dvy).astype('float32')
    out['speed_sum']=(df['speed_1']+df['speed_2']).astype('float32')
    out['speed_diff']=(df['speed_1']-df['speed_2']).astype('float32')
    out['acc_sum']=(df['acceleration_1']+df['acceleration_2']).astype('float32')
    out['orient_diff']=circ_diff(df['orientation_1'].values, df['orientation_2'].values).astype('float32')
    out['dir_diff']=circ_diff(df['direction_1'].values, df['direction_2'].values).astype('float32')
    out['dot_orient']=(df['ox_1']*df['ox_2']+df['oy_1']*df['oy_2']).astype('float32')
    out['is_ground']=(df['nfl_player_id_2']=='G').astype('int8')
    # projection onto line between players
    ux = dx/denom; uy = dy/denom
    out['p1_towards']=(df['vx_1']*ux+df['vy_1']*uy).astype('float32')
    out['p2_away']=(df['vx_2']*(-ux)+df['vy_2']*(-uy)).astype('float32')
    return out

def add_temporal(out, keys):
    # keys = ['game_play','nfl_player_id_1','nfl_player_id_2']
    base_cols = ['dist','closing_speed','speed_1','speed_2','rel_speed','speed_sum','acc_sum']
    g = out.groupby(keys, sort=False)
    for c in base_cols:
        for lag in (1,2,3,-1,-2,-3):
            out[f'{c}_lag{lag}']=g[c].shift(lag).astype('float32')
        r = g[c].rolling(5, min_periods=2)
        out[f'{c}_roll5_mean']=r.mean().reset_index(level=[0,1,2],drop=True).astype('float32')
        out[f'{c}_roll5_min']=r.min().reset_index(level=[0,1,2],drop=True).astype('float32')
    out['dist_min_lag']=out[[c for c in out.columns if c.startswith('dist_lag')]].min(axis=1).astype('float32')
    out['dist_d1']=(out['dist']-out['dist_lag1']).astype('float32')
    out['dist_d_1']=(out['dist']-out['dist_lag-1']).astype('float32')
    return out

def main():
    t0=time.time()
    tr = player_derived(load_tracking('train'))
    tr_test = player_derived(load_tracking('test'))
    results={}
    for split, trdf in [('train',tr),('test',tr_test)]:
        lab = build_labels(split)
        # remove pre-snap steps? keep >=0
        print(split,'labels',len(lab))
        m1=merge_side(lab,trdf,1)
        m2=merge_side(m1,trdf,2)
        f=pair_features(m2)
        keys=['game_play','nfl_player_id_1','nfl_player_id_2']
        order = m2[['game_play','nfl_player_id_1','nfl_player_id_2','step']].reset_index(drop=True)
        f=f.reset_index(drop=True)
        f=pd.concat([order, f],axis=1)
        if split=='train':
            f['contact']=m2['contact'].values
        f=f.sort_values(keys+['step']).reset_index(drop=True)
        f=add_temporal(f,keys)
        f=f.sort_values(['game_play','step']).reset_index(drop=True)
        f.to_parquet(os.path.join(OUT,f'{split}_track_feat.parquet'))
        print(split,'feat',f.shape,'time',round(time.time()-t0,1))
    print('done', round(time.time()-t0,1))

if __name__=='__main__':
    main()
