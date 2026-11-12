import os, sys, gc
import numpy as np
import pandas as pd

DATA = 'data'
OUT = 'workspaces/capella_pi'
os.makedirs(OUT, exist_ok=True)

TRK_COLS = ['game_play','nfl_player_id','step','team','position','jersey_number',
            'x_position','y_position','speed','distance','direction','orientation',
            'acceleration','sa']

def load_tracking():
    tr = pd.read_csv(f'{DATA}/train_player_tracking.csv', usecols=TRK_COLS)
    te = pd.read_csv(f'{DATA}/test_player_tracking.csv', usecols=TRK_COLS)
    tr['is_test'] = 0; te['is_test'] = 1
    trk = pd.concat([tr,te], ignore_index=True)
    trk['step'] = trk['step'].astype(np.int16)
    trk['nfl_player_id'] = trk['nfl_player_id'].astype(np.int32)
    # dedupe safety
    trk = trk.drop_duplicates(['game_play','step','nfl_player_id'])
    for c in ['team','position']:
        trk[c] = trk[c].astype('category')
    return trk

def build_helmet_step():
    cols=['game_play','view','frame','nfl_player_id','left','width','top','height']
    trh = pd.read_csv(f'{DATA}/train_baseline_helmets.csv', usecols=cols)
    teh = pd.read_csv(f'{DATA}/test_baseline_helmets.csv', usecols=cols)
    trh['is_test']=0; teh['is_test']=1
    h = pd.concat([trh,teh], ignore_index=True)
    del trh, teh; gc.collect()
    h['step'] = np.rint((h['frame']-300.0)/5.994).astype(np.int16)
    # some detections before snap -> keep only step>=-5
    h = h[h['step']>=-10]
    # box derived
    h['cx'] = (h['left']+h['width']/2.0).astype(np.float32)
    h['cy'] = (h['top']+h['height']/2.0).astype(np.float32)
    h['area'] = (h['width']*h['height']).astype(np.float32)
    h['diag'] = np.sqrt(h['width']**2+h['height']**2).astype(np.float32)
    h['left']=h['left'].astype(np.float32); h['top']=h['top'].astype(np.float32)
    h['width']=h['width'].astype(np.float32); h['height']=h['height'].astype(np.float32)
    h['nfl_player_id']=h['nfl_player_id'].astype(np.int32)
    # average per (game_play,view,step,player) in case multiple frames map to same step
    agg = h.groupby(['game_play','view','step','nfl_player_id'], observed=True).agg(
        cx=('cx','mean'), cy=('cy','mean'), left=('left','mean'), top=('top','mean'),
        width=('width','mean'), height=('height','mean'), area=('area','mean'),
        diag=('diag','mean')).reset_index()
    # pivot views
    out=None
    for view,pre in [('Sideline','s'),('Endzone','e')]:
        v=agg[agg['view']==view].drop(columns=['view'])
        v=v.rename(columns={c:f'{pre}_{c}' for c in ['cx','cy','left','top','width','height','area','diag']})
        out = v if out is None else out.merge(v, on=['game_play','step','nfl_player_id'], how='outer')
    return out

def add_person_feats(df, trk, idcol, prefix):
    t = trk.rename(columns={c: f'{prefix}_{c}' for c in trk.columns if c not in ('game_play','step','nfl_player_id')})
    return df.merge(t, left_on=['game_play','step',idcol], right_on=['game_play','step','nfl_player_id'], how='left').drop(columns=['nfl_player_id'])

def build_base(is_test=False):
    if is_test:
        ss = pd.read_csv(f'{DATA}/sample_submission.csv')
        parts = ss['contact_id'].str.split('_')
        df = pd.DataFrame({
            'game_play': parts.str[0]+'_'+parts.str[1],
            'step': parts.str[2].astype(np.int16),
            'nfl_player_id_1': parts.str[3].astype(np.int32),
            'nfl_player_id_2': parts.str[4],
        })
        df['contact']=np.nan
    else:
        df = pd.read_csv(f'{DATA}/train_labels.csv',
                         usecols=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact'])
        df['step']=df['step'].astype(np.int16)
        df['nfl_player_id_1']=df['nfl_player_id_1'].astype(np.int32)
    df['is_ground'] = (df['nfl_player_id_2']=='G').astype(np.int8)
    df['nfl_player_id_2_num'] = pd.to_numeric(df['nfl_player_id_2'], errors='coerce').astype('Int32')
    return df

def build_features(is_test=False):
    print('base...', flush=True)
    df = build_base(is_test)
    n0=len(df)
    print('base rows', n0, flush=True)
    def norm_id(s):
        s = s.copy()
        if s.dtype=='object':
            s = pd.to_numeric(s, errors='coerce')
        return s.astype(np.int32)
    df['p2id'] = df['nfl_player_id_2_num'].astype('float32')
    trk = load_tracking()
    trk['nfl_player_id']=trk['nfl_player_id'].astype(np.int32)
    print('merge tracking p1...', flush=True)
    df = add_person_feats(df, trk, 'nfl_player_id_1', 'p1')
    # p2 numeric
    df['p2_key'] = df['nfl_player_id_2_num'].astype('float64')
    t2 = trk.rename(columns={c: f'p2_{c}' for c in trk.columns if c not in ('game_play','step','nfl_player_id')})
    t2['nfl_player_id']=t2['nfl_player_id'].astype(np.int32)
    t2['nfl_player_id']=t2['nfl_player_id'].astype(np.float64)
    df['p2key']=df['nfl_player_id_2_num'].astype(np.float64)
    print('merge tracking p2...', flush=True)
    df = df.merge(t2, left_on=['game_play','step','p2key'],
                  right_on=['game_play','step','nfl_player_id'], how='left')
    df = df.drop(columns=['nfl_player_id'])
    del trk; gc.collect()

    eps=1e-6
    dx = df['p2_x_position']-df['p1_x_position']
    dy = df['p2_y_position']-df['p1_y_position']
    df['pair_dist'] = np.sqrt(dx*dx+dy*dy).astype(np.float32)
    df['pair_dx'] = np.abs(dx).astype(np.float32); df['pair_dy']=np.abs(dy).astype(np.float32)
    df['speed_sum']=df['p1_speed']+df['p2_speed']
    df['speed_diff']=df['p1_speed']-df['p2_speed']
    df['accel_sum']=df['p1_acceleration']+df['p2_acceleration']
    df['sa_diff']=df['p1_sa']-df['p2_sa']
    # velocity components
    r1=np.deg2rad(df['p1_direction']); r2=np.deg2rad(df['p2_direction'])
    vx1=df['p1_speed']*np.sin(r1); vy1=df['p1_speed']*np.cos(r1)
    vx2=df['p2_speed']*np.sin(r2); vy2=df['p2_speed']*np.cos(r2)
    df['closing_speed'] = (-(dx*(vx2-vx1)+dy*(vy2-vy1))/(df['pair_dist']+eps)).astype(np.float32)
    df['rel_vel'] = np.sqrt((vx2-vx1)**2+(vy2-vy1)**2).astype(np.float32)
    d_dir = np.abs(df['p1_direction']-df['p2_direction']); d_dir=np.minimum(d_dir,360-d_dir)
    df['dir_diff']=d_dir.astype(np.float32)
    o_dir = np.abs(df['p1_orientation']-df['p2_orientation']); o_dir=np.minimum(o_dir,360-o_dir)
    df['orient_diff']=o_dir.astype(np.float32)
    # same team
    df['same_team']=(df['p1_team'].astype(str)==df['p2_team'].astype(str)).astype(np.int8)
    df.loc[df['is_ground']==1,'same_team']=-1
    # time
    df['step']=df['step'].astype(np.int16)

    # ------- per-player context from player-player pairs -------
    print('context features...', flush=True)
    pp = df[df['is_ground']==0][['game_play','step','nfl_player_id_1','nfl_player_id_2_num','pair_dist']].copy()
    a = pp.rename(columns={'nfl_player_id_1':'pid'})[['game_play','step','pid','pair_dist']]
    b = pp.rename(columns={'nfl_player_id_2_num':'pid'})[['game_play','step','pid','pair_dist']]
    allp = pd.concat([a,b], ignore_index=True)
    allp['pid']=allp['pid'].astype(np.int32)
    allp['lt2']=(allp['pair_dist']<2).astype(np.int8)
    allp['lt4']=(allp['pair_dist']<4).astype(np.int8)
    ctx = allp.groupby(['game_play','step','pid']).agg(
        nn_dist=('pair_dist','min'),
        nn2=('lt2','sum'),
        nn4=('lt4','sum')).reset_index()
    ctx.columns=['game_play','step','pid','nn_dist','n_within_2','n_within_4']
    ctx['pid']=ctx['pid'].astype(np.int32)
    ctx['pid']=ctx['pid'].astype(np.float64)
    df['p1key']=df['nfl_player_id_1'].astype(np.float64)
    df=df.merge(ctx.rename(columns={c:f'p1_{c}' for c in ['nn_dist','n_within_2','n_within_4']}),
                left_on=['game_play','step','p1key'], right_on=['game_play','step','pid'], how='left').drop(columns=['pid'])
    df=df.merge(ctx.rename(columns={c:f'p2_{c}' for c in ['nn_dist','n_within_2','n_within_4']}),
                left_on=['game_play','step','p2key'], right_on=['game_play','step','pid'], how='left').drop(columns=['pid'])
    del pp,a,b,allp,ctx; gc.collect()

    # ------- helmet features -------
    print('helmet merge...', flush=True)
    hs = build_helmet_step()
    print('helmet-step rows', len(hs), flush=True)
    hs['nfl_player_id']=hs['nfl_player_id'].astype(np.int32)
    hcols=[c for c in hs.columns if c.startswith(('s_','e_'))]
    h1=hs.rename(columns={c:f'p1_{c}' for c in hcols})
    df=df.merge(h1, left_on=['game_play','step','nfl_player_id_1'], right_on=['game_play','step','nfl_player_id'], how='left').drop(columns=['nfl_player_id'])
    h2=hs.rename(columns={c:f'p2_{c}' for c in hcols})
    hs_id=hs['nfl_player_id'].astype(np.float64)
    h2=h2.drop(columns=['nfl_player_id']); h2['hid']=hs_id
    df=df.merge(h2, left_on=['game_play','step','p2key'], right_on=['game_play','step','hid'], how='left').drop(columns=['hid'])
    del hs,h1,h2; gc.collect()

    for v in ['s','e']:
        both = df[[f'p1_{v}_cx',f'p1_{v}_cy',f'p1_{v}_width',f'p1_{v}_height',
                   f'p2_{v}_cx',f'p2_{v}_cy',f'p2_{v}_width',f'p2_{v}_height']]
        present = both.notna().all(axis=1)
        cdx=df[f'p2_{v}_cx']-df[f'p1_{v}_cx']; cdy=df[f'p2_{v}_cy']-df[f'p1_{v}_cy']
        cdist=np.sqrt(cdx*cdx+cdy*cdy)
        scale=(df[f'p1_{v}_diag']+df[f'p2_{v}_diag'])/2.0+eps
        # IoU
        x1=np.maximum(df[f'p1_{v}_left'],df[f'p2_{v}_left'])
        y1=np.maximum(df[f'p1_{v}_top'],df[f'p2_{v}_top'])
        x2=np.minimum(df[f'p1_{v}_left']+df[f'p1_{v}_width'],df[f'p2_{v}_left']+df[f'p2_{v}_width'])
        y2=np.minimum(df[f'p1_{v}_top']+df[f'p1_{v}_height'],df[f'p2_{v}_top']+df[f'p2_{v}_height'])
        iw=np.clip(x2-x1,0,None); ih=np.clip(y2-y1,0,None); inter=iw*ih
        union=(df[f'p1_{v}_area']+df[f'p2_{v}_area']-inter)+eps
        df[f'{v}_cdist']=cdist.astype(np.float32)
        df[f'{v}_cdist_norm']=(cdist/scale).astype(np.float32)
        df[f'{v}_iou']=(inter/union).astype(np.float32)
        df[f'{v}_hgap']=(np.abs(cdx)/(df[f'p1_{v}_width']+df[f'p2_{v}_width'])/2+eps).astype(np.float32)
        df[f'{v}_vgap']=(np.abs(cdy)/(df[f'p1_{v}_height']+df[f'p2_{v}_height'])/2+eps).astype(np.float32)
        df[f'{v}_both']=present.astype(np.int8)
        df[f'{v}_p1_only']=(df[f'p1_{v}_cx'].notna() & df[f'p2_{v}_cx'].isna()).astype(np.int8)
        # lower helmet (ground proxy)
        df[f'{v}_p1_low']=(df[f'p1_{v}_cy']/ (df[f'p1_{v}_diag']+eps)).astype(np.float32)
    df['h_min_cdist']=df[['s_cdist','e_cdist']].min(axis=1).astype(np.float32)
    df['h_min_cdist_norm']=df[['s_cdist_norm','e_cdist_norm']].min(axis=1).astype(np.float32)
    df['h_max_iou']=df[['s_iou','e_iou']].max(axis=1).astype(np.float32)
    df['h_views_both']=(df['s_both']+df['e_both']).astype(np.int8)

    # ------- temporal features on key signals -------
    print('temporal...', flush=True)
    keys=['game_play','nfl_player_id_1','nfl_player_id_2']
    df=df.sort_values(keys+['step']).reset_index(drop=True)
    grp=df.groupby(keys, sort=False, observed=True)
    base_signals=['pair_dist','closing_speed','h_min_cdist_norm','h_max_iou',
                  'p1_speed','p2_speed','p1_acceleration','p2_acceleration','p1_sa','p2_sa',
                  'p1_nn_dist','p2_nn_dist']
    gb=df.groupby(keys, sort=False, observed=True)
    for c in base_signals:
        if c not in df: continue
        g=gb[c]
        df[f'{c}_lag1']=g.shift(1).astype(np.float32)
        df[f'{c}_lead1']=g.shift(-1).astype(np.float32)
        df[f'{c}_lag2']=g.shift(2).astype(np.float32)
        df[f'{c}_lead2']=g.shift(-2).astype(np.float32)
    rollcols=[c for c in ['pair_dist','closing_speed','p1_speed','p2_speed','h_max_iou','p1_acceleration','p2_acceleration'] if c in df]
    for c in rollcols:
        r5=gb[c].rolling(5,center=True,min_periods=1).mean()
        r5.index=r5.index.droplevel([0,1,2])
        df[f'{c}_roll5']=r5.reindex(df.index).astype(np.float32)
        r11=gb[c].rolling(11,center=True,min_periods=1).max()
        r11.index=r11.index.droplevel([0,1,2])
        df[f'{c}_roll11max']=r11.reindex(df.index).astype(np.float32)
    df['pair_dist_diff']=(df['pair_dist']-df['pair_dist_lag1']).astype(np.float32)

    # drop raw id columns & helpers
    drop=['contact_id'] if 'contact_id' in df else []
    drop += ['nfl_player_id_2_num','p1key','p2key','p2id']
    df=df.drop(columns=drop, errors='ignore')
    return df

if __name__=='__main__':
    which=sys.argv[1]
    if which in ('train','both'):
        df=build_features(False)
        df.to_parquet(f'{OUT}/train_feat.parquet')
        print('saved train', df.shape, flush=True)
        del df; gc.collect()
    if which in ('test','both'):
        df=build_features(True)
        df.to_parquet(f'{OUT}/test_feat.parquet')
        print('saved test', df.shape, flush=True)
