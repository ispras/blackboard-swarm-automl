import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LSTMNet, TransformerNet, TCNNet, predict, SEQ


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--cache', default='workspaces/sirius_tau/cache')
    p.add_argument('--model', default='lstm')
    p.add_argument('--hidden', type=int, default=384)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--drop', type=float, default=0.1)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--nf_dim', type=int, default=256)
    return p.parse_args()


def main():
    a = get_args()
    dev = 'cuda'
    outdir = a.run
    names = open(f'{a.cache}/feature_names.txt').read().split('\n')
    uoi = names.index('u_out')
    X = np.load(f'{a.cache}/X_train.npy', mmap_mode='r')
    trn_idx = np.load(f'{a.cache}/trn_idx.npy')
    val_idx = np.load(f'{a.cache}/val_idx.npy')
    y = np.load(f'{a.cache}/y_train.npy')
    Xtr = np.asarray(X[trn_idx], dtype=np.float32)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    del Xtr
    Xva = (np.asarray(X[val_idx], dtype=np.float32) - mean) / std
    uova = np.asarray(X[val_idx, :, uoi])
    yva = np.asarray(y[val_idx], dtype=np.float32)
    nf = len(names)
    if a.model in ('lstm', 'gru'):
        model = LSTMNet(nf, a.hidden, a.layers, a.drop, a.model).to(dev)
    elif a.model == 'tcn':
        model = TCNNet(nf, a.hidden, a.layers, a.drop).to(dev)
    else:
        model = TransformerNet(nf, a.hidden, a.layers, a.drop, a.heads, a.nf_dim).to(dev)
    model.load_state_dict(torch.load(f'{outdir}/model.pt', map_location=dev))
    pv = predict(model, torch.from_numpy(Xva).to(dev))
    m_insp = float(np.abs(pv - yva)[uova == 0].mean())
    m_all = float(np.abs(pv - yva).mean())
    print(f'val_insp={m_insp:.5f} val_all={m_all:.5f}')
    np.save(f'{outdir}/val_pred.npy', pv.astype(np.float32))

    Xte = (np.asarray(np.load(f'{a.cache}/X_test.npy', mmap_mode='r'), dtype=np.float32) - mean) / std
    pt = predict(model, torch.from_numpy(Xte).to(dev))
    tbid = np.load(f'{a.cache}/test_breath_ids.npy')
    te = pd.read_csv('data/test.csv', usecols=['id', 'breath_id', 'time_step'])
    te_sorted = te.sort_values(['breath_id', 'time_step'], kind='mergesort').reset_index(drop=True)
    key_bid = np.repeat(tbid, SEQ)
    assert len(te_sorted) == len(key_bid)
    assert (te_sorted['breath_id'].values == key_bid).all()
    sub = pd.DataFrame({'id': te_sorted['id'].values, 'pressure': pt.reshape(-1)})
    sample = pd.read_csv('data/sample_submission.csv')
    sub = sample[['id']].merge(sub, on='id', how='left')
    assert sub['pressure'].notna().all()
    sub.to_csv(f'{outdir}/test_pred.csv', index=False)
    val_idx_saved = np.load(f'{outdir}/val_idx.npy') if os.path.exists(f'{outdir}/val_idx.npy') else val_idx
    np.save(f'{outdir}/val_idx.npy', val_idx_saved)
    rj = f'{outdir}/result.json'
    d = json.load(open(rj)) if os.path.exists(rj) else {}
    d.update({'val_insp': m_insp, 'val_all': m_all})
    json.dump(d, open(rj, 'w'), indent=2)
    print('wrote', outdir)


if __name__ == '__main__':
    main()
