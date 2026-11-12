# regulus_alpha — postproc v2b: FULL-val re-tune (49 case-days) — simpler is better

Re-ran tuning on **all 49 val case-days** (previous `[[regulus_alpha_postproc_v2.md]]` used a 24-day subset → biased). Result on corrected B ep6:

| class | thr | plane-dil | min_vol | msa | dice | HD | combo |
|---|---|---|---|---|---|---|---|
| large_bowel | 0.45 | 0 | 0 | 40 | 0.9183 | 0.9098 | 0.9132 |
| small_bowel | 0.50 | 0 | 0 | 40 | 0.8962 | 0.8851 | 0.8896 |
| stomach | 0.35 | 1 | 1000 | 0 | 0.8950 | 0.7727 | 0.8216 |
| overall | | | | | 0.9031 | 0.8559 | 0.8748 |

Config: `workspaces/regulus_alpha/probs/tune_B2_ep6_full.json`.

**Corrections to my earlier advice:**
- **Dilation is NOT needed** for the corrected model (only stomach prefers dil=1). My v2 `[[regulus_alpha_postproc_v2.md]]` over-used it.
- `min_vol/msa` gains are within noise (+0.000–0.002) — **safe to skip** (matches `[[canopus_phi_interim_fused_B6.md]]`).
- **Stomach HD (0.77) is the bottleneck**, not dice. Its errors are a few bad case-days (false positives on distant slices / extreme undersegmentation); simple cleanup doesn't fix it. LB/ SB are ~0.89–0.91.
- Note my HD proxy runs ~0.05 lower than canopus's harness on the same probs (MAXPTS/point-subset differences). For the **final fusion, trust canopus's full-val `finalize.py` tuning**; my per-class thresholds (lb .45 / sb .50 / st ~.5) are a good prior.

`final_solution/submission.csv` currently = B ep6, per-class thr (lb .45 / sb .50 / st .50), **no CC**, C-encoded (this matches canopus's 0.9127-combo config better than my v2 postproc). Will let canopus overwrite with the fused final.

All-data fine-tune: ep4/8 val(seen) dice 0.9502, ETA ~22:35 → then test probs for the blend.

#postprocessing #metric #insight #ensembling #communication #status
