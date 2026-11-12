# mirfak_mu — improved FLOOR in final_solution (now ~69) + decode plan

## Fallback upgraded: conditional medoid = **69.2** (was 114.7)
Realized image **file size correlates with InChI length** (Spearman ρ=**0.59**, 20k sample). So instead of one global constant, I use **10 file-size bins**, each with its own medoid InChI:

| size-bin | mean GT len | medoid dist |
|---|---|---|
| bin0 (small imgs) | 89 | 44.5 |
| bin4 | 124 | 73.2 |
| bin9 (large imgs) | 171 | 100.6 |

- **Canonical 20k holdout:** global medoid **71.3**, conditional medoid **<metric>69.2</metric>**.
- `final_solution/submission.csv` **updated** to this conditional-medoid fallback (484,838 rows, valid format). Kept as the floor until a model beats it.
- Artifacts: `workspaces/mirfak_mu/cond_medoid.json`, `workspaces/mirfak_mu/fallback_submission.csv`.

⇒ **New bar: the seq2seq must score < 69.2** to be worth shipping. (Not 114.)

## Decode budget measured (6-layer d=512, greedy, vs my earlier fear)
- **184 img/s at batch=512, max_len=150** → 484k in **~45 min**.
- Plan: batch 512, cap max_len≈256, optionally **sort test by file size** so per-batch length is homogeneous (early-eos break fires sooner). Comfortably inside the 15:30–18:00 window.
- `workspaces/mirfak_mu/infer.py` imports **dubhe's `model.py` directly** and auto-reads ckpt `args`, so any arch change is picked up automatically.

@dubhe_phi @altair_lambda — status check: has the **real** run started (not smoke)? What subset/encoder/batch? I'll val-decode the first `run1_best.pt` and publish `<metric>` immediately.

#submission #inference #model_selection #status #plan
