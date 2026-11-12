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


def load(kind):
    base = pd.read_parquet(f'{kind}_pair.parquet')
    hel = pd.read_parquet(f'../achernar_sigma/{kind}_helmet_features.parquet')
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

y = tr.contact.values.astype('int8')
oof = np.zeros(len(tr), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
rounds = []
for f in range(NFOLD):
    trn = tr[tr.fold != f]; val = tr[tr.fold == f]
    dtr = lgb.Dataset(trn[fc], label=trn.contact.values, categorical_feature=cats)
    dva = lgb.Dataset(val[fc], label=val.contact.values, categorical_feature=cats, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[val.index.values] = m.predict(val[fc], num_iteration=m.best_iteration).astype('float32')
    test_pred += m.predict(te[fc], num_iteration=m.best_iteration).astype('float32') / NFOLD
    rounds.append(m.best_iteration)
    log('fold', f, 'iter', m.best_iteration, 'auc', round(roc_auc_score(val.contact, oof[val.index.values]), 5))
    del dtr, dva, m; gc.collect()

out = tr[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2','contact','fold']].copy()
out['prob'] = oof
out.to_parquet('oof_pair_h.parquet', index=False)
tout = te[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']].copy()
tout['prob'] = test_pred.astype('float32')
tout.to_parquet('testpred_pair_h.parquet', index=False)
auc = roc_auc_score(out.contact, out.prob)
best = (-1, 1.0)
for t in np.unique(np.quantile(out.prob, np.linspace(0.9, 0.9995, 300))):
    mm = matthews_corrcoef(out.contact, out.prob >= t)
    if mm > best[0]: best = (mm, t)
log('DONE auc', round(auc,5), 'mcc', round(best[0],5), 'thr', best[1], 'rounds', rounds)
json.dump({'auc': float(auc), 'mcc': float(best[0]), 'thr': float(best[1]), 'rounds': rounds},
          open('pair_h_result.json','w'), indent=2)
