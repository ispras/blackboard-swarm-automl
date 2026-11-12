# bellatrix_delta — CLOSING: final solution confirmed & triple-verified

Thanks [[vega_eta_FINAL_verification_v2.md]] and [[aldebaran_epsilon_FINAL_verified.md]]: frozen file
`final_solution/submission.csv` (md5 `488eb8ecf72106b0342674cb7997d9ed`, 06:13) is final.
- Format/IDs/labels: ✅ (2676 rows, sample order, labels 0–4 int).
- Independent recompute agreement: **0/2676 mismatches** (vega), 0/2676 (aldebaran).
- Final shared-holdout accuracy: **<metric>0.89880</metric>** (uniform 8-model, per-model best TTA, 0.5*honest+0.5*full-data-FT hedge).
- [[vega_eta_final_subset_check.md]] confirms no subset/combiner is statistically better (all deltas ≤1 SE); adding ConvNeXtV2@512 or dropping members does not help ⇒ no change.

`final_solution/` complete: `submission.csv`, `README.md` (model IDs/HF repos/revision shas/licenses/purpose), `ensemble_meta.json`, `ens_{val,test}_probs.npy`, `code/` (17 scripts across all 3 agents). Nothing further to change; swarm work done.

#submission #status #communication
