import numpy as np
import pandas as pd
import numba

TRACK_COLS = ['game_play','nfl_player_id','step','x_position','y_position','speed',
              'distance','direction','orientation','acceleration','sa','team','position']


def load_tracking(path):
    t = pd.read_csv(path, usecols=TRACK_COLS)
    for c in ['x_position','y_position','speed','distance','direction','orientation','acceleration','sa']:
        t[c] = t[c].astype('float32')
    t['step'] = t['step'].astype('int32')
    t['nfl_player_id'] = t['nfl_player_id'].astype('int64')
    t['is_home'] = (t['team'].str.lower().str[0] == 'h').astype('int8')
    return t


def prep_keys(labels):
    labels = labels.copy()
    labels['step'] = labels['step'].astype('int32')
    labels['nfl_player_id_1'] = labels['nfl_player_id_1'].astype('int64')
    labels['is_ground'] = (labels['nfl_player_id_2'].astype(str) == 'G').astype('int8')
    labels['nfl_player_id_2'] = pd.to_numeric(labels['nfl_player_id_2'], errors='coerce').fillna(-1).astype('int64')
    return labels


def merge_track(labels, track):
    p = track.rename(columns={
        'nfl_player_id':'nfl_player_id_1','x_position':'x1','y_position':'y1','speed':'speed1',
        'distance':'distance1','direction':'direction1','orientation':'orientation1',
        'acceleration':'accel1','sa':'sa1','team':'team1','position':'pos1'})
    p = p.drop(columns=['is_home'])
    labels = labels.merge(p, on=['game_play','nfl_player_id_1','step'], how='left')

    q = track.rename(columns={
        'nfl_player_id':'nfl_player_id_2','x_position':'x2','y_position':'y2','speed':'speed2',
        'distance':'distance2','direction':'direction2','orientation':'orientation2',
        'acceleration':'accel2','sa':'sa2','team':'team2','position':'pos2','is_home':'home2'})
    labels = labels.merge(q, on=['game_play','nfl_player_id_2','step'], how='left')
    return labels


@numba.njit(cache=True)
def _lag(vals, gstart, s):
    n = len(vals)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        j = i - s
        if j >= gstart[i]:
            out[i] = vals[j]
    return out


@numba.njit(cache=True)
def _roll(vals, gstart, w, mode):
    # mode 0=min 1=max 2=mean (past window incl current, min_periods=1)
    n = len(vals)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lo = i - w + 1
        if lo < gstart[i]:
            lo = gstart[i]
        c = 0
        if mode == 0:
            best = np.inf
        elif mode == 1:
            best = -np.inf
        else:
            best = 0.0
        for j in range(lo, i + 1):
            v = vals[j]
            if not np.isnan(v):
                c += 1
                if mode == 0:
                    if v < best: best = v
                elif mode == 1:
                    if v > best: best = v
                else:
                    best += v
        if c == 0:
            out[i] = np.nan
        elif mode == 2:
            out[i] = best / c
        else:
            out[i] = best
    return out


def _gstart_from_keys(df):
    gp = df['game_play'].values
    p1 = df['nfl_player_id_1'].values
    p2 = df['nfl_player_id_2'].values
    n = len(df)
    gs = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(1, n):
        if gp[i] != gp[i-1] or p1[i] != p1[i-1] or p2[i] != p2[i-1]:
            cur = i
        gs[i] = cur
    return gs


def _sort_and_gstart(df):
    df = df.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step'], kind='mergesort').reset_index(drop=True)
    return df, _gstart_from_keys(df)


def add_temporal(df, cfgs):
    df, gstart = _sort_and_gstart(df)
    for col, lags, rmin, rmean in cfgs:
        if col not in df.columns:
            continue
        vals = df[col].values.astype('float32')
        for s in lags:
            df[f'{col}_lag{s}'] = _lag(vals, gstart, s)
        for w in rmin:
            df[f'{col}_rmin{w}'] = _roll(vals, gstart, w, 0)
        for w in rmean:
            df[f'{col}_rmean{w}'] = _roll(vals, gstart, w, 2)
    return df


def add_basic(df):
    dx = (df['x1'] - df['x2']).values
    dy = (df['y1'] - df['y2']).values
    dist = np.sqrt(dx*dx + dy*dy)
    df['dx'] = dx.astype('float32'); df['dy'] = dy.astype('float32')
    df['dist'] = dist.astype('float32')
    df['dist_x'] = np.abs(dx).astype('float32'); df['dist_y'] = np.abs(dy).astype('float32')

    rad1 = np.deg2rad(df['direction1'].values)
    rad2 = np.deg2rad(df['direction2'].values)
    v1x = df['speed1'].values * np.sin(rad1); v1y = df['speed1'].values * np.cos(rad1)
    v2x = df['speed2'].values * np.sin(rad2); v2y = df['speed2'].values * np.cos(rad2)
    rvx = v1x - v2x; rvy = v1y - v2y
    df['rel_speed'] = np.sqrt(rvx*rvx + rvy*rvy).astype('float32')
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = dx / dist; uy = dy / dist
        closing = -(rvx*ux + rvy*uy)
    df['closing_speed'] = np.nan_to_num(closing, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
    df['speed_diff'] = (df['speed1'] - df['speed2']).astype('float32')
    df['speed_sum'] = (df['speed1'] + df['speed2']).astype('float32')
    df['accel_sum'] = (df['accel1'] + df['accel2']).astype('float32')
    df['sa_diff'] = (df['sa1'] - df['sa2']).astype('float32')

    dd = (df['direction1'] - df['direction2']).values % 360.0
    df['dir_diff'] = np.minimum(dd, 360.0-dd).astype('float32')
    od = (df['orientation1'] - df['orientation2']).values % 360.0
    df['orient_diff'] = np.minimum(od, 360.0-od).astype('float32')

    with np.errstate(divide='ignore', invalid='ignore'):
        ttc = np.where(df['closing_speed'].values > 0.05, dist / df['closing_speed'].values, 99.0)
    df['ttc'] = np.clip(ttc, 0, 99).astype('float32')
    df['same_team'] = (df['team1'] == df['team2']).astype('int8')
    return df


def add_step_global(df):
    d = df[['game_play','step','dist','speed1','x1','y1']].copy()
    d['c1'] = (d['dist'] < 1.0).astype('float32')
    d['c2'] = (d['dist'] < 2.0).astype('float32')
    d['c3'] = (d['dist'] < 3.0).astype('float32')
    agg = d.groupby(['game_play','step']).agg(
        n_close1=('c1','sum'), n_close2=('c2','sum'), n_close3=('c3','sum'),
        step_min_dist=('dist','min'), step_mean_dist=('dist','mean'),
        step_mean_speed=('speed1','mean'), step_max_speed=('speed1','max')).reset_index()
    for c in agg.columns[2:]:
        agg[c] = agg[c].astype('float32')
    return df.merge(agg, on=['game_play','step'], how='left')


PAIR_CFGS = [
    ('dist', (-10,-5,-3,-1,1,3,5,10), (3,5,10), (5,10)),
    ('closing_speed', (-3,-1,1,3), (), (5,)),
    ('rel_speed', (-3,3), (), (5,)),
    ('speed1', (-5,-1,1,5), (), (5,)),
    ('speed2', (-5,-1,1,5), (), (5,)),
    ('accel1', (-1,1), (), ()),
    ('accel2', (-1,1), (), ()),
    ('sa1', (-1,1), (), ()),
    ('sa2', (-1,1), (), ()),
    ('ttc', (-1,1), (5,), ()),
    ('dir_diff', (-1,1), (), ()),
]

GROUND_CFGS = [
    ('speed1', (-10,-5,-3,-1,1,3,5,10), (3,5,10), (5,10)),
    ('accel1', (-5,-3,-1,1,3,5), (5,), (5,)),
    ('sa1', (-3,-1,1,3), (), (5,)),
    ('distance1', (-3,-1,1,3), (5,), (5,)),
    ('direction1', (-3,3), (), ()),
]


def build_pair(nonground, track):
    df = add_basic(merge_track(nonground, track))
    df = add_temporal(df, PAIR_CFGS)
    df = add_step_global(df)
    return df


def build_ground(ground, track):
    p = track.rename(columns={
        'nfl_player_id':'nfl_player_id_1','x_position':'x1','y_position':'y1','speed':'speed1',
        'distance':'distance1','direction':'direction1','orientation':'orientation1',
        'acceleration':'accel1','sa':'sa1','team':'team1','position':'pos1'})
    df = ground.merge(p, on=['game_play','nfl_player_id_1','step'], how='left')
    df['nfl_player_id_2'] = -1
    df = add_temporal(df, GROUND_CFGS)
    # step globals from full tracking per step
    g = track.groupby(['game_play','step']).agg(
        g_step_mean_speed=('speed','mean'), g_step_max_speed=('speed','max'),
        g_step_mean_accel=('acceleration','mean'), g_step_min_speed=('speed','min')).reset_index()
    for c in g.columns[2:]:
        g[c] = g[c].astype('float32')
    df = df.merge(g, on=['game_play','step'], how='left')
    return df


def add_pos_codes(dfs, pos_col):
    vals = pd.concat([d[pos_col] for d in dfs if pos_col in d.columns]).astype(str).unique()
    mp = {v:i for i,v in enumerate(sorted(vals))}
    for d in dfs:
        if pos_col in d.columns:
            d[pos_col+'_code'] = d[pos_col].astype(str).map(mp).fillna(-1).astype('int16')
    return mp
