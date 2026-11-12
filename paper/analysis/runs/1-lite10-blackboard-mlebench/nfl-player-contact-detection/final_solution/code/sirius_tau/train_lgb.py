import pandas as pd, numpy as np, lightgbm as lgb, sys, os, json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef

OUT='workspaces/sirius_tau'

def best_thr(y, p):
    order = np.argsort(-p)
    ys = y[order].astype(np.float64); ps = p[order].astype(np.float64)
    P = np.cumsum(ys)
    totalP = P[-1]
    N_total = len(ys) - totalP
    TP = P
    FP = (np.arange(len(ys))+1.0) - P
    FN = totalP - TP
    TN = N_total - FP
    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = np.where(denom>0,(TP*TN-FP*FN)/denom,0.0)
    k = int(np.argmax(mcc))
    return float(ps[k]), float(mcc[k])

def main():
    df = pd.read_parquet(f'{OUT}/feat_train.parquet')
    test = pd.read_parquet(f'{OUT}/feat_test.parquet')

    cat_feats = ['position_1','position_2','team_1','team_2']
    for c in cat_feats:
        df[c] = df[c].astype('category')
        test[c] = pd.Categorical(test[c], categories=df[c].cat.categories)

    drop = ['contact_id','game_play','datetime','step','nfl_player_id_1','nfl_player_id_2','contact']
    feats = [c for c in df.columns if c not in drop]
    print('n feats', len(feats), feats)

    y = df['contact'].values.astype(int)
    groups = df['game_play'].values
    X = df[feats]

    params = dict(objective='binary', learning_rate=0.05, num_leaves=128,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
                  num_threads=16)
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(df))
    best_iters = []
    for i,(tri,vai) in enumerate(gkf.split(X,y,groups)):
        dtr = lgb.Dataset(X.iloc[tri], y[tri], categorical_feature=cat_feats)
        dva = lgb.Dataset(X.iloc[vai], y[vai], categorical_feature=cat_feats)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(200)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        b,mcc = best_thr(y[vai], oof[vai])
        print(f'fold {i} best_iter {m.best_iteration} thr {b:.4f} mcc {mcc:.4f}')
    thr, mcc = best_thr(y, oof)
    print(f'OOF thr {thr:.4f} MCC {mcc:.5f}')
    np.save(f'{OUT}/oof_lgb.npy', oof)

    # full model
    it = int(np.mean(best_iters)*1.1)+1
    dall = lgb.Dataset(X, y, categorical_feature=cat_feats)
    mfull = lgb.train(params, dall, num_boost_round=it)
    pt = mfull.predict(test[feats])
    np.save(f'{OUT}/test_lgb.npy', pt)
    sub = pd.read_csv('data/sample_submission.csv')
    sub['contact'] = (pt>=thr).astype(int)
    os.makedirs('final_solution',exist_ok=True)
    sub.to_csv('final_solution/submission.csv', index=False)
    print('saved submission, pos rate', sub.contact.mean(), 'full_iters', it)
    json.dump({'oof_mcc':float(mcc),'thr':float(thr),'n_feats':len(feats),'iters':it,
               'feats':feats}, open(f'{OUT}/lgb_meta.json','w'))

if __name__=='__main__':
    main()
