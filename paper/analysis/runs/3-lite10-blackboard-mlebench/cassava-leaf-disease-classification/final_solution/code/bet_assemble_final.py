import os, json
import numpy as np, pandas as pd

ROOT = '/download_job_code/payload/sandbox'
OUT = os.path.join(ROOT, 'final_solution')

# Each member: name -> (honest_val, honest_test, ft_test_or_None, val_ids_or_None)
MEMBERS = {
    'bellatrix_efficientnet_b5_512': (
        'workspaces/bellatrix_delta/run_b5_512/tta4/val_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_b5_512/tta4/test_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_b5_512/tta4/test_probs_ft_tta4.npy', None),
    'bellatrix_efficientnet_b5_640': (
        'workspaces/bellatrix_delta/run_b5_640/tta4ms/val_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_b5_640/tta4ms/test_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_b5_640/tta4ms/test_probs_ft_tta4.npy', None),
    'bellatrix_convnext_base_384': (
        'workspaces/bellatrix_delta/run_cnxt_384/tta4ms/val_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_cnxt_384/tta4ms/test_probs_tta4.npy',
        'workspaces/bellatrix_delta/run_cnxt_384/tta4ms/test_probs_ft_tta4.npy', None),
    'vega_eva02_base_448': (
        'workspaces/vega_eta/runs/eva02b448_f0_s42/val_probs_tta.npy',
        'workspaces/vega_eta/runs/eva02b448_f0_s42/test_probs.npy',
        'workspaces/vega_eta/runs/eva02b448_f0_s42/full/test_probs.npy', None),
    'vega_caformer_b36_384': (
        'workspaces/vega_eta/runs/caformer384_f0_s42/val_probs_tta.npy',
        'workspaces/vega_eta/runs/caformer384_f0_s42/test_probs.npy',
        'workspaces/vega_eta/runs/caformer384_f0_s42/full/test_probs.npy', None),
    'aldebaran_swin_base_384': (
        'workspaces/aldebaran_epsilon/swin/val_probs.npy',
        'workspaces/aldebaran_epsilon/swin/test_probs.npy',
        'workspaces/aldebaran_epsilon/swin_ft/test_probs_ft.npy',
        'workspaces/aldebaran_epsilon/swin/val_ids.csv'),
    'aldebaran_vit_base_384': (
        'workspaces/aldebaran_epsilon/vit/tta4/val_probs_tta4.npy',
        'workspaces/aldebaran_epsilon/vit/tta4/test_probs_tta4.npy',
        'workspaces/aldebaran_epsilon/vit_ft/tta4/test_probs_tta4.npy',
        'workspaces/aldebaran_epsilon/vit/val_ids.csv'),
    'vega_convnextv2_base_384': (
        'workspaces/vega_eta/runs/cnv2b384_f0_s42/val_probs_tta.npy',
        'workspaces/vega_eta/runs/cnv2b384_f0_s42/test_probs.npy',
        'workspaces/vega_eta/runs/cnv2b384_f0_s42/full/test_probs.npy', None),
}


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None); return p / p.sum(1, keepdims=True)


def main():
    cv = pd.read_csv(os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    va = cv[cv.fold == 0].reset_index(drop=True); y = va.label.values; va_ids = va.image_id.values
    sample = pd.read_csv(os.path.join(ROOT, 'data/sample_submission.csv'))
    names, V, T, meta = [], [], [], {}
    for name, (vp, tp, ftp, vi) in MEMBERS.items():
        v = norm(np.load(os.path.join(ROOT, vp)))
        if vi:
            ids = pd.read_csv(os.path.join(ROOT, vi)).image_id.values
            if len(ids) == len(v) and not np.array_equal(ids, va_ids):
                order = pd.Index(ids); v = v[[order.get_loc(i) for i in va_ids]]
        assert len(v) == len(va), (name, len(v))
        h = norm(np.load(os.path.join(ROOT, tp)))
        assert len(h) == len(sample), name
        if ftp and os.path.exists(os.path.join(ROOT, ftp)):
            f = norm(np.load(os.path.join(ROOT, ftp)))
            agree = float((h.argmax(1) == f.argmax(1)).mean())
            t = 0.5 * h + 0.5 * f
            ft_used = True
        else:
            agree = 1.0; t = h; ft_used = False
        acc = float((v.argmax(1) == y).mean())
        names.append(name); V.append(v); T.append(t)
        meta[name] = {'val_acc': acc, 'ft_used': ft_used, 'honest_vs_ft_test_agreement': agree}
        print(f'{name:34s} val={acc:.5f} ft={ft_used} agree={agree:.4f}')

    V = np.stack(V); T = np.stack(T)
    ens_val = V.mean(0); ens_test = T.mean(0)
    fold0 = float((ens_val.argmax(1) == y).mean())
    print(f'\nUNIFORM {len(names)}-MODEL HEDGED ENSEMBLE fold0 accuracy = {fold0:.5f}')

    sub = pd.DataFrame({'image_id': sample.image_id, 'label': ens_test.argmax(1)})
    sub.to_csv(os.path.join(OUT, 'submission.csv'), index=False)
    np.save(os.path.join(OUT, 'ens_val_probs.npy'), ens_val)
    np.save(os.path.join(OUT, 'ens_test_probs.npy'), ens_test)
    out = {'n_models': len(names), 'members': meta, 'fold0_accuracy_uniform_hedged': fold0,
           'weights': 'uniform', 'test_label_distribution': {int(k): int(v) for k, v in sub.label.value_counts().sort_index().items()}}
    json.dump(out, open(os.path.join(OUT, 'ensemble_meta.json'), 'w'), indent=2)
    print('wrote', os.path.join(OUT, 'submission.csv'))
    print('label dist', out['test_label_distribution'])


if __name__ == '__main__':
    main()
