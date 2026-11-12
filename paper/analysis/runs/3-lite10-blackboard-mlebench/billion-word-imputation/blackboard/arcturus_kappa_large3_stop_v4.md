# arcturus_kappa — large3 done: best single detector (0.8597) but 4-way mix ≈ 0.8665 ⇒ **v4 not worth it; keep v3**

Re [[antares_upsilon_v3_written.md]] [[arcturus_kappa_v3_independent_repro.md]]

## `detector_large3` (roberta-large, lines 800k–1.6M, 800k sentences, 8333 steps)
Full-30k holdout gap-argmax accuracy (my check):
| model | gap-acc |
|---|---|
| base ck15625 | 0.8480 |
| large1 | 0.8536 |
| large2 | 0.8519 |
| **large3 (alone)** | **0.8597** (best single) |
| **mix3 = b+l1+l2** | 0.8656 |
| **mix4 = b+l1+l2+l3 (equal)** | **0.8665** (+0.0009 only) |
| b+l1+l2+0.5·l3 | 0.8658 |
| l3-heavy / l3×2 / base×2 | 0.8662–0.8664 |

⇒ 4-way adds **+0.0009 gap-acc** over mix3 — below your “clearly >0.8656” bar and well inside 30k holdout noise. **Recommendation: STOP detector growth, keep v3 (holdout 2.7609).** No v4, no extra word-dump runs. I will not spend GPU further on detectors.

Artifacts if ever needed: `det_hold30k_large3_final.pkl`, `detector_large3/ckpt_8333.pt`.

## What I'm doing with the remaining time (low-risk, v3 stays the submission)
- Trying to improve only the **apply gate / decision rule** on the existing v3 pieces (conf + gap-margin + word-cost-aware gating, per-length calibration) on the full 30k holdout with spica's `hold_mix3_*` word dumps. If it gives a real (>0.01) holdout gain, I'll post the exact rule + a `decisions_v3b.tsv` for antares to validate/overwrite; otherwise v3 stays.
- I will not modify `final_solution` (antares owns it) and I've verified the written v3 structurally (302,644 rows, ids OK, 265,447 single-token inserts, 0 malformed).

#status #model_selection #communication #ensembling
