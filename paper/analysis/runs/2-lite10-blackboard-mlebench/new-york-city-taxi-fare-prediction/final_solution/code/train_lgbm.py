import os, sys, time, argparse, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_io as D
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=0.25)
    ap.add_argument('--leaves', type=int, default=255)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--rounds', type=int, default=2500)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--tag', type=str, default='lgb_v1')
    ap.add_argument('--eval_max', type=float, default=100.0,
                    help='trim fare above this for early-stopping validation')
    ap.add_argument('--min_leaf', type=int, default=200)
    ap.add_argument('--linear', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    X, y, Xh, yh, hv = D.load(frac=args.frac)
    print('benchmark head', {c: float(X[:1000, i].mean()) for i, c in enumerate(D.FEATURES[:3])}, flush=True)

    params = dict(
        objective='regression', metric='rmse', learning_rate=args.lr,
        num_leaves=args.leaves, min_data_in_leaf=args.min_leaf, feature_fraction=0.85,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
        num_threads=args.threads, verbose=-1, seed=42,
    )
    if args.linear:
        params['linear_tree'] = True
        params['linear_lambda'] = 1.0
    trim = hv & (yh >= 0.5) & (yh <= args.eval_max)
    dtrain = lgb.Dataset(X, label=y, feature_name=D.FEATURES_ALL, free_raw_data=True)
    dvalid = lgb.Dataset(Xh[trim], label=yh[trim], reference=dtrain, free_raw_data=True)
    evals = {}
    model = lgb.train(params, dtrain, num_boost_round=args.rounds,
                      valid_sets=[dvalid], valid_names=['hold_trim'],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100),
                                 lgb.record_evaluation(evals)])
    model.save_model(os.path.join(BASE, f'{args.tag}.txt'))
    print('best_iter', model.best_iteration, 'best_score', model.best_score)

    ph = model.predict(Xh, num_iteration=model.best_iteration)
    ph_v = ph[hv]
    yh_v = yh[hv]
    r_all = D.rmse(yh, ph)
    r_v = D.rmse(yh_v, ph_v)
    r_trim = D.rmse(yh[trim], ph[trim])
    print(f'HOLDOUT_RMSE_valid={r_v:.5f}  HOLDOUT_RMSE_all={r_all:.5f}  RMSE_trim(0.5,{args.eval_max})={r_trim:.5f}', flush=True)

    keys = pd.read_csv(D.HOLD_KEYS)['key'].values
    pd.DataFrame({'key': keys, 'fare_amount': yh, 'pred': ph}).to_csv(
        os.path.join(BASE, f'{args.tag}_holdout_pred.csv'), index=False)

    tkeys, Xt = D.load_test()
    pt = model.predict(Xt, num_iteration=model.best_iteration)
    sub = pd.DataFrame({'key': tkeys, 'fare_amount': pt})
    sub.to_csv(os.path.join(BASE, f'{args.tag}_submission.csv'), index=False)
    print(sub['fare_amount'].describe())

    json.dump({'tag': args.tag, 'frac': args.frac, 'best_iter': model.best_iteration,
               'holdout_rmse_valid': r_v, 'holdout_rmse_all': r_all, 'holdout_rmse_trim': r_trim,
               'eval_max': args.eval_max,
               'leaves': args.leaves, 'lr': args.lr, 'elapsed': time.time()-t0},
              open(os.path.join(BASE, f'{args.tag}_meta.json'), 'w'), indent=2)
    print('DONE', time.time()-t0)

if __name__ == '__main__':
    main()
