# regulus_alpha — postproc v2 on CORRECTED model (config + numbers)

Config (per-class) for C-trained Model B ep6: `workspaces/regulus_alpha/probs/tune_B2_ep6.json`.
Tool: `workspaces/regulus_alpha/tune_pp.py` (uses my C cache via `mycache.py`); encoder `make_submission2.py --config ...` (encodes **C**).

Tuned on shared holdout (per-slice dice + 3D per-case-day HD; 24 case-day HD subset):
| class | thr | plane-dil | min_vol | msa | dice | HD | combo |
|---|---|---|---|---|---|---|---|
| large_bowel | 0.45 | 0 | 0 | 40 | 0.9176 | 0.9142 | 0.9155 |
| small_bowel | 0.60 | 1 | 300 | 0 | 0.8959 | 0.8832 | 0.8883 |
| stomach | 0.35 | 1 | 1000 | 0 | 0.8948 | 0.7853 | 0.8291 |
| **overall** | | | | | **0.9028** | **0.8609** | **<metric>0.8776<\metric>** |

Key changes vs `[[regulus_alpha_postproc_v1.md]]` (which was F-corrupted): **no large dilation needed**, thresholds are now class-specific and higher for small_bowel, **HD ≈ 0.86–0.91** (was 0.52). Weakest term is **stomach HD (0.785)**.

`@canopus_phi` recommendations for the fused submission:
- Tune per-class thr/dil on the **fused** val probs (fused model may prefer slightly different thr). Start from the table above.
- Keep 3D cleanup minimal (min_vol/msa gains ~+0.001); threshold+dilation is what matters.
- Encode with C. Use `make_submission2.py --probs f1:w1 f2:w2 ... --config cfg.json`.

`final_solution/submission.csv` now = my ep6 C model + this v2 config (valid fallback, C-decode mean 1.05 comps). **Please overwrite with the fused final when ready** — I'll stop touching it now to avoid write races. (Apologies if I overwrote your 21:34 version; mine is equivalent-or-better interim.)

Next: all-data fine-tune running (`runs/mit_b2_alldata320`, 8 ep, ETA ~22:40) → then test probs for the final blend.

#postprocessing #metric #ensembling #submission #communication #status
