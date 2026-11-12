import os, time, json, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef

t0 = time.time(); NJOBS = 12; NFOLD = 5


def log(*a): print(f'[{time.time()-t0:.1f}s]', *a, flush=True)


EXCL = {'contact_id','game_play','nfl_player_id_1','nfl_player_id_2','contact',
        'game_key','is_ground','step','fold'}
EXTRA = ['nn_any_1','nn_opp_1','nn_team_1','n_within2_1','n_opp_within2_1','n_within4_1',
         'speed_nn_opp_1','nn_any_2','nn_opp_2','nn_team_2','n_within2_2','n_opp_within2_2',
         'n_within4_2','speed_nn_opp_2','same_pos','toward1','toward2','is_nn1','is_nn2',
         'is_nn_opp1','is_nn_opp2','dist_rmin3','dist_rmin7','dist_rmean5','closing_rmean5',
         'past_close_5','speed_max','speed_min','speed_absdiff','acc_sum','sa_sum',
         'sa_absdiff','cum_dist_1','cum_dist_2','orient_dir_diff_1','orient_dir_diff_2',
         'speed_rmin5_1','speed_rmin5_2']

folds = pd.read_csv('../castor_alpha/folds.csv')
fmap = dict(zip(folds.game_play, folds.fold))
PARAMS = dict(objective='binary', metric='auc', learning_rate=0.06, num_leaves=96,
              min_child_samples=60, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=2.0, n_jobs=NJOBS, verbose=-1, max_bin=127, seed=777)


OLD_HELM = ['contact_id', 'Endzone_dist', 'Endzone_ndist', 'Endzone_dx', 'Endzone_dy', 'Endzone_adx_n', 'Endzone_ady_n', 'Endzone_iou', 'Endzone_sratio', 'Endzone_smsum', 'Endzone_pair_present', 'Endzone_cy', 'Endzone_top', 'Endzone_bottom', 'Endzone_height', 'Endzone_width', 'Endzone_area', 'Endzone_cy_rank', 'Endzone_cy_rel', 'Endzone_size_rel', 'Endzone_cy_to_max', 'Endzone_bottom_to_max', 'Endzone_p1_present', 'Sideline_dist', 'Sideline_ndist', 'Sideline_dx', 'Sideline_dy', 'Sideline_adx_n', 'Sideline_ady_n', 'Sideline_iou', 'Sideline_sratio', 'Sideline_smsum', 'Sideline_pair_present', 'Sideline_cy', 'Sideline_top', 'Sideline_bottom', 'Sideline_height', 'Sideline_width', 'Sideline_area', 'Sideline_cy_rank', 'Sideline_cy_rel', 'Sideline_size_rel', 'Sideline_cy_to_max', 'Sideline_bottom_to_max', 'Sideline_p1_present', 'x_ndist_min', 'x_ndist_max', 'x_ndist_mean', 'x_nviews', 'x_dist_mean', 'x_cy_rank_mean', 'x_ndist_min_lag1', 'x_ndist_min_lead1', 'x_ndist_min_d1', 'Endzone_ndist_lag1', 'Endzone_ndist_lead1', 'Endzone_ndist_d1', 'Sideline_ndist_lag1', 'Sideline_ndist_lead1', 'Sideline_ndist_d1', 'x_ndist_rmin5', 'x_ndist_rmean5', 'x_cy_rank_d1', 'x_cy_rank_rmax5', 'x_cy_rank_rmean5']
def load(kind):
    base = pd.read_parquet(f'{kind}_pair.parquet')
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(f'../achernar_sigma/{kind}_helmet_features.parquet').schema.names)
    hcols = [c for c in OLD_HELM if c in avail]
    hel = pd.read_parquet(f'../achernar_sigma/{kind}_helmet_features.parquet', columns=hcols)
    trk = pd.read_parquet(f'../achernar_sigma/{kind}_tracking_features.parquet',
                          columns=['contact_id'] + EXTRA)
    hel = hel.drop(columns=[c for c in ['contact'] if c in hel.columns])
    df = base.merge(hel, on='contact_id', how='left').merge(trk, on='contact_id', how='left')
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype('float32')
    df['game_key'] = df['game_play'].str.split('_').str[0].astype('int32')
    df['fold'] = df['game_play'].map(fmap)
    return df


tr = load('train'); gc.collect(); log('train merged', tr.shape)
te = load('test'); gc.collect(); log('test merged', te.shape)
fc = [c for c in tr.columns if c not in EXCL and tr[c].dtype != object]
cats = ['pos1_code'] if 'pos1_code' in fc else []
log('nfeat', len(fc))

rounds_fixed=[1470,565,1316,1965,1210]
test_pred = np.zeros(len(te), dtype=np.float32)
for f in range(NFOLD):
    trn = tr[tr.fold != f]
    dtr = lgb.Dataset(trn[fc], label=trn.contact.values, categorical_feature=cats)
    m = lgb.train(PARAMS, dtr, num_boost_round=rounds_fixed[f])
    test_pred += m.predict(te[fc], num_iteration=rounds_fixed[f]).astype('float32') / NFOLD
    log('fold', f, 'done', rounds_fixed[f])
    del dtr, m; gc.collect()
tout = te[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']].copy()
tout['prob'] = test_pred.astype('float32')
tout.to_parquet('testpred_pair_h.parquet', index=False)
log('saved fresh testpred_pair_h.parquet')

