import pandas as pd, numpy as np, sys, os

DATA='data'
OUT='workspaces/sirius_tau'

TCOL = ['game_play','nfl_player_id','step','x_position','y_position','speed',
        'distance','direction','orientation','acceleration','sa','position','team']

def load_tracking(name):
    tr = pd.read_csv(f'{DATA}/{name}_player_tracking.csv')
    tr = tr[TCOL].copy()
    tr['nfl_player_id'] = tr['nfl_player_id'].astype(str)
    tr['step'] = tr['step'].astype(int)
    return tr

def build(labels_path, is_test):
    if is_test:
        ss = pd.read_csv(f'{DATA}/sample_submission.csv')
        parts = ss.contact_id.str.rsplit('_', n=3, expand=True)
        lb = pd.DataFrame({
            'game_play': parts[0],
            'step': parts[1].astype(int),
            'nfl_player_id_1': parts[2],
            'nfl_player_id_2': parts[3],
        })
        lb['contact'] = -1
    else:
        lb = pd.read_csv(labels_path)
        lb['step'] = lb['step'].astype(int)
        lb['nfl_player_id_1'] = lb['nfl_player_id_1'].astype(str)
        lb['nfl_player_id_2'] = lb['nfl_player_id_2'].astype(str)

    tr = load_tracking('test' if is_test else 'train')

    p1 = tr.rename(columns={c: c+'_1' for c in tr.columns if c not in ('game_play','step','nfl_player_id')})
    p2 = tr.rename(columns={c: c+'_2' for c in tr.columns if c not in ('game_play','step','nfl_player_id')})

    df = lb.merge(p1, left_on=['game_play','step','nfl_player_id_1'], right_on=['game_play','step','nfl_player_id'], how='left')
    df = df.drop(columns=['nfl_player_id'])
    df = df.merge(p2, left_on=['game_play','step','nfl_player_id_2'], right_on=['game_play','step','nfl_player_id'], how='left')
    df = df.drop(columns=['nfl_player_id'])
    return df

def add_feats(df):
    df['is_ground'] = (df['nfl_player_id_2'] == 'G').astype(np.int8)
    x1,y1,x2,y2 = df.x_position_1, df.y_position_1, df.x_position_2, df.y_position_2
    df['dx'] = x1-x2
    df['dy'] = y1-y2
    df['dist'] = np.sqrt(df.dx**2+df.dy**2)
    df.loc[df.is_ground==1,'dist'] = np.nan
    # speed product / diff
    df['speed_sum'] = df.speed_1 + df.speed_2
    df['speed_diff'] = (df.speed_1 - df.speed_2).abs()
    df['speed_prod'] = df.speed_1*df.speed_2
    df['acc_sum'] = df.acceleration_1 + df.acceleration_2
    df['sa_sum'] = df.sa_1 + df.sa_2
    # velocity dot/cos
    d1 = np.deg2rad(df.direction_1); d2 = np.deg2rad(df.direction_2)
    v1x = df.speed_1*np.sin(d1); v1y = df.speed_1*np.cos(d1)
    v2x = df.speed_2*np.sin(d2); v2y = df.speed_2*np.cos(d2)
    df['rel_speed'] = np.sqrt((v1x-v2x)**2+(v1y-v2y)**2)
    df['cos_dir'] = (v1x*v2x+v1y*v2y)/(df.speed_1*df.speed_2+1e-6)
    df['dir_diff'] = np.abs((df.direction_1-df.direction_2+180)%360-180)
    df['orient_diff'] = np.abs((df.orientation_1-df.orientation_2+180)%360-180)
    # closing speed: rate of change of dist along line
    with np.errstate(invalid='ignore',divide='ignore'):
        ux = df.dx/df.dist; uy = df.dy/df.dist
    df['closing'] = (v1x-v2x)*ux + (v1y-v2y)*uy
    df.loc[df.is_ground==1,['cos_dir','rel_speed','closing','speed_diff','speed_prod','dir_diff','orient_diff']] = np.nan
    return df

def add_temporal(df):
    df = df.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).reset_index(drop=True)
    g = df.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'], sort=False)
    for col in ['dist','speed_1','speed_2','closing']:
        for k in [-2,-1,1,2]:
            df[f'{col}_sh{k}'] = g[col].shift(-k)
    df['dist_delta'] = df['dist'] - df['dist_sh-1']
    df['dist_delta_f'] = df['dist_sh1'] - df['dist']
    df['dist_min_w'] = g['dist'].transform(lambda s: s.rolling(5,center=True,min_periods=1).min())
    df['dist_mean_w'] = g['dist'].transform(lambda s: s.rolling(5,center=True,min_periods=1).mean())
    df['dist_max_w'] = g['dist'].transform(lambda s: s.rolling(5,center=True,min_periods=1).max())
    df['speed1_mean_w'] = g['speed_1'].transform(lambda s: s.rolling(5,center=True,min_periods=1).mean())
    df['speed2_mean_w'] = g['speed_2'].transform(lambda s: s.rolling(5,center=True,min_periods=1).mean())
    df['speed1_max_w'] = g['speed_1'].transform(lambda s: s.rolling(9,center=True,min_periods=1).max())
    df['speed2_max_w'] = g['speed_2'].transform(lambda s: s.rolling(9,center=True,min_periods=1).max())
    g2 = df.groupby(['game_play','nfl_player_id_1'], sort=False)
    df['p1_speed_prev'] = g2['speed_1'].shift(1)
    df['p1_speed_next'] = g2['speed_1'].shift(-1)
    return df

if __name__=='__main__':
    which = sys.argv[1]
    labels_path = None if which=='test' else f'{DATA}/train_labels.csv'
    df = build(labels_path, is_test=(which=='test'))
    df = add_feats(df)
    df = add_temporal(df)
    df.to_parquet(f'{OUT}/feat_{which}.tmp.parquet')
    os.replace(f'{OUT}/feat_{which}.tmp.parquet', f'{OUT}/feat_{which}.parquet')
    print(which, df.shape)
    print(df[['dist','speed_sum','closing']].describe())
