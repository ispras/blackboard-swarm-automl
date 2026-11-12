import os, time, gc
import numpy as np
import pandas as pd
import build_track_features as B

DATA = '../../data'
OUT = '.'
t0 = time.time()


def log(*a):
    print(f'[{time.time()-t0:.1f}s]', *a, flush=True)


def parse_test(sub):
    parts = sub['contact_id'].str.split('_')
    out = pd.DataFrame({
        'game_play': parts.str[0] + '_' + parts.str[1],
        'step': parts.str[2].astype('int32'),
        'nfl_player_id_1': parts.str[3].astype('int64'),
        'nfl_player_id_2': parts.str[4],
    })
    out['is_ground'] = (out['nfl_player_id_2'] == 'G').astype('int8')
    out['nfl_player_id_2'] = pd.to_numeric(out['nfl_player_id_2'], errors='coerce').fillna(-1).astype('int64')
    out['contact'] = 0
    return out


def finalize(df, name):
    # add game_key
    df['game_key'] = df['game_play'].str.split('_').str[0].astype('int32')
    drop = ['datetime', 'team1', 'pos1', 'team2', 'pos2', 'home2']
    df = df.drop(columns=[c for c in drop if c in df.columns])
    if 'pos1_code' not in df.columns and 'pos1' in df.columns:
        pass
    df.to_parquet(os.path.join(OUT, name), index=False)
    log('saved', name, df.shape)
    return df


# ---------------- TRAIN ----------------
log('loading labels + tracking')
labels = B.prep_keys(pd.read_csv(f'{DATA}/train_labels.csv'))
track = B.load_tracking(f'{DATA}/train_player_tracking.csv')
log('loaded', labels.shape, track.shape)

ng = labels[labels.is_ground == 0].copy()
gr = labels[labels.is_ground == 1].copy()
del labels
gc.collect()

pair = B.build_pair(ng, track)
log('pair built', pair.shape)
del ng
gc.collect()

ground = B.build_ground(gr, track)
log('ground built', ground.shape)
del gr
gc.collect()

mp = B.add_pos_codes([pair, ground], 'pos1')
import json
json.dump(mp, open('pos_map.json','w'))
pair = finalize(pair, 'train_pair.parquet')
ground = finalize(ground, 'train_ground.parquet')
del pair, ground
gc.collect()

# ---------------- TEST ----------------
log('loading test')
sub = pd.read_csv(f'{DATA}/sample_submission.csv')
te = parse_test(sub)
track_te = B.load_tracking(f'{DATA}/test_player_tracking.csv')
log('test keys', te.shape, track_te.shape)

te_ng = te[te.is_ground == 0].copy()
te_gr = te[te.is_ground == 1].copy()
del te
gc.collect()

pair_te = B.build_pair(te_ng, track_te)
log('test pair built', pair_te.shape)
del te_ng
gc.collect()
ground_te = B.build_ground(te_gr, track_te)
log('test ground built', ground_te.shape)
del te_gr
gc.collect()

import json
mp = json.load(open('pos_map.json'))
for d in (pair_te, ground_te):
    d['pos1_code'] = d['pos1'].astype(str).map(mp).fillna(-1).astype('int16')
pair_te = finalize(pair_te, 'test_pair.parquet')
ground_te = finalize(ground_te, 'test_ground.parquet')
log('done')
