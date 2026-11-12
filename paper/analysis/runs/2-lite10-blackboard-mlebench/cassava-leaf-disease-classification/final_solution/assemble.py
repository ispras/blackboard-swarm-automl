import os, glob, argparse
import numpy as np, pandas as pd

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', default='')
    a = p.parse_args()
    out = a.out or a.run
    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    df['fold'] = -1
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, vi) in enumerate(skf.split(df, df.label)):
        df.loc[vi, 'fold'] = f

    oof = np.zeros((len(df), 5), dtype=np.float32)
    got = np.zeros(len(df), dtype=bool)
    for f in range(5):
        fp = os.path.join(a.run, f'val_probs_fold{f}.npy')
        if os.path.exists(fp):
            vp = np.load(fp)
            vi = df.index[df.fold == f].values
            oof[vi] = vp
            got[vi] = True
    print('oof coverage', got.mean())
    acc = (oof[got].argmax(1) == df.label.values[got]).mean()
    print('OOF acc', acc)
    o = pd.DataFrame(oof, columns=[f'p{i}' for i in range(5)])
    o.insert(0, 'image_id', df.image_id.values)
    o['label'] = df.label.values
    o.to_csv(os.path.join(out, 'oof_probs.csv'), index=False)

    test = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]
    tps = []
    for f in range(5):
        fp = os.path.join(a.run, f'test_probs_fold{f}.npy')
        if os.path.exists(fp):
            tps.append(np.load(fp))
    tp = np.mean(tps, axis=0)
    print('test folds used', len(tps))
    t = pd.DataFrame(tp, columns=[f'p{i}' for i in range(5)])
    t.insert(0, 'image_id', test.image_id.values)
    t.to_csv(os.path.join(out, 'test_probs.csv'), index=False)
    sub = pd.DataFrame({'image_id': test.image_id.values, 'label': tp.argmax(1)})
    sub.to_csv(os.path.join(out, 'submission.csv'), index=False)
    print(sub.label.value_counts().to_dict())


if __name__ == '__main__':
    main()
