"""Build dn3_max: raw BirdNET v2.4 per-species max probability over 45 windows.

BirdNET v2.4 was run by pollux_epsilon on spectrally-denoised audio at 3 denoise strengths
(alpha = 1.5, 2.5, 3.5), 15 windows of 3 s (stride 0.5 s) per 10 s clip.  For each species we take
max over all windows and all denoise strengths.  No training, no labels.

Inputs (workspaces/pollux_epsilon):
  birdnet_denoised.npz.npy    (322, 15, 19) logits, alpha 1.5
  birdnet_denoised_a2.5.npy   (322, 15, 19) logits, alpha 2.5
  birdnet_denoised_a3.5.npy   (322, 15, 19) logits, alpha 3.5
Outputs: oof_dn3_max.npy (258,19, rows = sorted train rec_ids), test_dn3_max.npy (64,19)
"""
import numpy as np

R = '/download_job_code/payload/sandbox'
EPS = f'{R}/workspaces/pollux_epsilon'
files = ['birdnet_denoised.npz.npy', 'birdnet_denoised_a2.5.npy', 'birdnet_denoised_a3.5.npy']
probs = [1 / (1 + np.exp(-np.load(f'{EPS}/{f}'))) for f in files]   # (322,15,19) each
dn3 = np.concatenate(probs, axis=1).max(1)                          # (322,19)
# real split indices
import pandas as pd
cv = pd.read_csv(f'{R}/data/essential_data/CVfolds_2.txt')
tr = np.array(sorted(cv[cv.fold == 0].rec_id.tolist()))
te = np.array(sorted(cv[cv.fold == 1].rec_id.tolist()))
np.save(f'{R}/workspaces/pollux_beta/preds/oof_dn3_max.npy', dn3[tr].astype(np.float32))
np.save(f'{R}/workspaces/pollux_beta/preds/test_dn3_max.npy', dn3[te].astype(np.float32))
print('saved dn3_max', dn3[tr].shape, dn3[te].shape)
