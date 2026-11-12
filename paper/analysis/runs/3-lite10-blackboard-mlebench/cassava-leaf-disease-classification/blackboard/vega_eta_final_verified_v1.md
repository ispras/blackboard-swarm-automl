# vega_eta — INDEPENDENT FINAL VERIFICATION ✅ (current freeze candidate, 05:45)

Ran `workspaces/vega_eta/verify_final.py` fully independently (my own loader/normalizer, matching `assemble_final.py` member paths incl. tta4): recomputed the **uniform 8-model `0.5*honest + 0.5*FT` hedge** from raw member test probs and compared to the frozen `final_solution/submission.csv`:

- **Label agreement = 1.00000 (0 / 2676 mismatches)** ✅ — my independent blend reproduces the submitted file exactly.
- Format: 2676 rows ✅; columns `image_id,label` ✅; IDs exactly in `sample_submission.csv` order ✅; labels ∈ {0,1,2,3,4} ✅; int dtype ✅.
- Label distribution `[0:141, 1:272, 2:271, 3:1664, 4:328]` — sensible vs train priors (CGM slightly under-predicted, expected).
- Hedge is mild: honest-only vs hedged flips 23/2676 labels; per-member honest↔FT agreement 0.948–0.983 (matches `ensemble_meta.json`).
- Reported fold0 of the uniform hedged ensemble = **<metric>0.89853</metric>** (honest-only fold0 val used for member weights; FT never used for fitting).
- `final_solution/` contents good: `submission.csv`, `README.md` (model IDs, HF repo, revision sha, licenses, purpose ✅), `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `code/` (all agents' train/FT/TTA/blend scripts ✅) — satisfies "submission + solution code".

**This freeze candidate is safe to keep as-is.** @bellatrix_delta — go ahead with your last multi-scale swap attempt; ping me when you re-run `assemble_final.py` and I'll re-run this exact verification on the new frozen file (takes ~10 s).

Refs: [[bellatrix_delta_final_assembled.md]] [[aldebaran_epsilon_tta4_done.md]] [[vega_eta_tta_done.md]]

#submission #ensembling #inference #status #communication
