import os, sys, time, argparse, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_io as D

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--device', type=str, default='CPU')
    ap.add_argument('--eval_max', type=float, default=100.0)
    ap.add_argument('--tag', type=str, default='cat_v1')
    args = ap.parse_args()

    import catboost as cb
    t0 = time.time()
    X, y, Xh, yh, hv = D.load(frac=args.frac)
    trim = hv & (yh >= 0.5) & (yh <= args.eval_max)
    train_pool = cb.Pool(X, label=y, feature_names=D.FEATURES_ALL)
    valid_pool = cb.Pool(Xh[trim], label=yh[trim], feature_names=D.FEATURES_ALL)

    params = dict(loss_function='RMSE', iterations=args.iters, learning_rate=args.lr,
                  depth=args.depth, l2_leaf_reg=4.0, random_seed=42,
                  border_count=128, task_type=args.device, thread_count=args.threads,
                  od_type='Iter', od_wait=120, verbose=100, allow_writing_files=False)
    if args.device.upper() == 'GPU':
        params['devices'] = '0'
    model = cb.CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(os.path.join(BASE, f'{args.tag}.cbm'))
    print('best_iter', model.get_best_iteration(), 'best_score', model.get_best_score())

    ph = model.predict(Xh)
    r_v = D.rmse(yh[hv], ph[hv])
    r_all = D.rmse(yh, ph)
    r_trim = D.rmse(yh[trim], ph[trim])
    print(f'CAT HOLDOUT valid={r_v:.5f} all={r_all:.5f} trim={r_trim:.5f}', flush=True)

    keys = pd.read_csv(D.HOLD_KEYS)['key'].values
    pd.DataFrame({'key': keys, 'fare_amount': yh, 'pred': ph}).to_csv(
        os.path.join(BASE, f'{args.tag}_holdout_pred.csv'), index=False)
    tkeys, Xt = D.load_test()
    pt = model.predict(Xt)
    pd.DataFrame({'key': tkeys, 'fare_amount': pt}).to_csv(
        os.path.join(BASE, f'{args.tag}_submission.csv'), index=False)
    print(pd.Series(pt).describe())
    json.dump({'tag': args.tag, 'frac': args.frac, 'best_iter': int(model.get_best_iteration()),
               'holdout_rmse_valid': r_v, 'holdout_rmse_all': r_all, 'holdout_rmse_trim': r_trim,
               'depth': args.depth, 'lr': args.lr, 'device': args.device, 'elapsed': time.time()-t0},
              open(os.path.join(BASE, f'{args.tag}_meta.json'), 'w'), indent=2)
    print('DONE', time.time()-t0)

if __name__ == '__main__':
    main()
