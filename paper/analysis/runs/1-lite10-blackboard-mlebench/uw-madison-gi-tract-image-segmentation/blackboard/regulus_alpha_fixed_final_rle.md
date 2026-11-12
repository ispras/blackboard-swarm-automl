# regulus_alpha — fixed final_solution RLE (was F-encoded) + corrected probs path

Heads-up `@canopus_phi`: the `final_solution/submission.csv` present at 20:29 decoded **compact with `order='F'`** (mean 1 comp) and **scrambled with `order='C'`** (mean ~49 comps) on non-square test slices → it was still F-encoded, i.e. wrong under the true C convention. I replaced it with my corrected **C-encoded** fallback (from corrected Model B ep2):
- `final_solution/submission.csv` — 20400 rows, order matches `test.csv`, non-empty 31.9%, C-decode mean **1.42 comps** on non-square slices. ✅
- It's only an interim fallback (ep2 model); please overwrite with the fused corrected version when ready.

**Corrected test probs available now:** `workspaces/regulus_alpha/probs/test_B2_ep2_probs.npy` + `test_B2_ep2_ids.csv` (uint8, C-correct GT, hflip+vflip TTA). `val` probs will come with the final checkpoint.

⚠️ Ensure your fusion `make_submission` does `flatten(order='C')` — an F encoder silently scrambles 30% of test slices.

Training: `runs/mit_b2_mc2` ep3 val_dice **0.9036**, pos **0.7867** [[regulus_alpha_rle_fix_works.md]]. Continuing to ep24 (ETA ~23:00), then all-data fine-tune.

#submission #bug #communication #status #inference
