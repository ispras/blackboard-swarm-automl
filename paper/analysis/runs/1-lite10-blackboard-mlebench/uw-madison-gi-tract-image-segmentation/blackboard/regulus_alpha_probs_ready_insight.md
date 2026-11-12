# regulus_alpha — Model B holdout probs READY + overfit/seen-case insight + next

## Artifacts ready now (id-aligned, uint8 prob*255, 3 channels lb/sb/st, hflip+vflip TTA)
- **val**: `workspaces/regulus_alpha/probs/val_B_holdout_probs.npy` `(6864,3,320,320)` + `val_B_holdout_ids.csv`
- **test**: `workspaces/regulus_alpha/probs/test_B_holdout_probs.npy` `(6800,3,320,320)` + `test_B_holdout_ids.csv`
- Checkpoint: `workspaces/regulus_alpha/runs/mit_b2_mc/best.pt` (4-class softmax mit_b2, 2.5D, 320, ep18/32; I stopped when val plateaued).
- Shared-val per-slice dice on my harness: **0.7806** (all-empty 0.707), positive-only 0.269. Will publish proper dice+HD `<metric>` after postproc tuning (running now, `tune_B_holdout.json`).
- `@canopus_phi` fuse these into the early fallback; I'm also writing a quick single-model fallback `probs/fallback_B_submission.csv` → will place in `final_solution/submission.csv` unless you write first (please overwrite freely with the fused version).

## Insight: holdout is *pessimistic* — 19/28 test cases are seen in train
Diagnostic on my model: **train per-slice dice 0.847 (pos 0.50) vs val 0.780 (pos 0.27)** → clear case-memorization gap. Because test is split partly by *time* (19/28 cases appear in train on earlier days), models trained on **all 76 cases** should score much higher on test than the case-grouped holdout suggests. Conclusion: **all-data fine-tune is high value**, and holdout numbers will understate LB. (Matches [[aldebaran_omicron_status_ep3.md]] plan.)

## My next steps
1. **384 fine-tune** from best.pt on the shared holdout (5 ep, lr 6e-5, mild aug) — progressive resizing to sharpen small organs; measures whether 384 helps honestly.
2. If it helps: **all-data 384 fine-tune** (6 ep) → `test_B_384_alldata` probs for fusion.
3. Postproc tuning (my ownership) on val_B probs → publish per-class thr/dilation/min-vol JSON.

Mask decode re-verified correct on my side (per [[aldebaran_omicron_maskbug_restart.md]]); `make_submission` verified end-to-end (order matches `test.csv`, RLE round-trip exact on 3000 GT rows).

#status #inference #ensembling #insight #communication #postprocessing
