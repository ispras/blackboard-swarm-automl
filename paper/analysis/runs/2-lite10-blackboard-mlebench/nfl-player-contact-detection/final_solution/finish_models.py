import os, time, json, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score

t0 = time.time()
NJOBS = 12
NFOLD = 5


def log(*a):
    print(f'[{time.time()-t0:.1f}s]', *a, flush=True)


EXCL = {'contact_id','game_play','nfl_player_id_1','nfl_player_id_2','contact',
        'game_key','is_ground','step','fold'}
folds = pd.read_csv('../castor_alpha/folds.csv')
fmap = dict(zip(folds.game_play, folds.fold))
PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05,
              num_leaves=128, min_child_samples=80, feature_fraction=0.75,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              n_jobs=NJOBS, verbose=-1, max_bin=255, seed=42)


def feat_cols(df):
    return [c for c in df.columns if c not in EXCL and df[c].dtype != object]


def run_ground():
    tr = pd.read_parquet('train_ground.parquet')
    te = pd.read_parquet('test_ground.parquet')
    tr['fold'] = tr['game_play'].map(fmap)
    fc = feat_cols(tr)
    cats = ['pos1_code'] if 'pos1_code' in fc else []
    log('ground', tr.shape, len(fc))
    oof = np.zeros(len(tr), dtype=np.float32)
    test_pred = np.zeros(len(te), dtype=np.float32)
    rounds = []
    for f in range(NFOLD):
        trn = tr[tr.fold != f]; val = tr[tr.fold == f]
        dtr = lgb.Dataset(trn[fc], label=trn.contact.values, categorical_feature=cats)
        dva = lgb.Dataset(val[fc], label=val.contact.values, categorical_feature=cats, reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[val.index.values] = m.predict(val[fc], num_iteration=m.best_iteration).astype('float32')
        test_pred += m.predict(te[fc], num_iteration=m.best_iteration).astype('float32') / NFOLD
        rounds.append(m.best_iteration)
        log('ground fold', f, m.best_iteration, round(roc_auc_score(val.contact, oof[val.index.values]), 5))
        del dtr, dva, m; gc.collect()
    out = tr[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2','contact','fold']].copy()
    out['prob'] = oof
    out.to_parquet('oof_ground.parquet', index=False)
    tout = te[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']].copy()
    tout['prob'] = test_pred.astype('float32')
    tout.to_parquet('testpred_ground.parquet', index=False)
    log('ground done', rounds, 'AUC', round(roc_auc_score(out.contact, out.prob), 5))


def run_pair_testpred():
    tr = pd.read_parquet('train_pair.parquet')
    te = pd.read_parquet('test_pair.parquet')
    tr['fold'] = tr['game_play'].map(fmap)
    fc = feat_cols(tr)
    cats = ['pos1_code'] if 'pos1_code' in fc else []
    rounds = [900, 500, 700, 800, 200]
    log('pair testpred', tr.shape, len(fc))
    test_pred = np.zeros(len(te), dtype=np.float32)
    for f in range(NFOLD):
        trn = tr[tr.fold != f]
        dtr = lgb.Dataset(trn[fc], label=trn.contact.values, categorical_feature=cats)
        m = lgb.train(PARAMS, dtr, num_boost_round=rounds[f])
        test_pred += m.predict(te[fc], num_iteration=rounds[f]).astype('float32') / NFOLD
        log('pair fold', f, 'done', rounds[f])
        del dtr, m; gc.collect()
    tout = te[['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']].copy()
    tout['prob'] = test_pred.astype('float32')
    tout.to_parquet('testpred_pair.parquet', index=False)
    log('pair testpred saved')


def summarize():
    op = pd.read_parquet('oof_pair.parquet')
    og = pd.read_parquet('oof_ground.parquet')
    res = {'auc_pair': float(roc_auc_score(op.contact, op.prob)),
           'auc_ground': float(roc_auc_score(og.contact, og.prob))}
    for name, d in [('pair', op), ('ground', og)]:
        best = (-1, 1.0)
        for t in np.unique(np.quantile(d.prob, np.linspace(0.5, 0.9995, 500))):
            m = matthews_corrcoef(d.contact, d.prob >= t)
            if m > best[0]: best = (m, t)
        res[f'mcc_{name}'] = float(best[0]); res[f'thr_{name}'] = float(best[1])
        log(name, 'best MCC', round(best[0],5), best[1])
    json.dump(res, open('cv_result.json','w'), indent=2)
    log(res)


if __name__ == '__main__':
    run_ground()
    run_pair_testpred()
    summarize()
    # quick submission at 0.5 for fallback
    op = pd.read_parquet('oof_pair.parquet')
    # find thresholds from oof quickly
    bp = max(((matthews_corrcoef(op.contact, op.prob >= t), t) for t in np.unique(np.quantile(op.prob, np.linspace(0.9,0.999,100)))))
    og = pd.read_parquet('oof_ground.parquet')
    bg = max(((matthews_corrcoef(og.contact, og.prob >= t), t) for t in np.unique(np.quantile(og.prob, np.linspace(0.9,0.999,100)))))
    tp = pd.read_parquet('testpred_pair.parquet'); tg = pd.read_parquet('testpred_ground.parquet')
    sub = pd.read_csv('../../data/sample_submission.csv')
    pr = pd.concat([tp[['contact_id','prob']], tg[['contact_id','prob']]])
    sub = sub.merge(pr, on='contact_id', how='left')
    thr = np.where(sub.contact_id.str.endswith('_G'), bg[1], bp[1])
    sub['contact'] = (sub.prob.fillna(0) >= thr).astype(int)
    sub[['contact_id','contact']].to_csv('submission_lgb_v1.csv', index=False)
    log('fallback submission', sub.contact.mean(), 'thr', bp, bg)
