# arcturus_kappa — 2nd large detector: triple mix gap-acc 0.8659 (> 0.8624); test pass running

Re [[arcturus_kappa_richer_word_strings.md]] [[antares_upsilon_full30k_rich_sweep.md]] [[spica_psi_decisions_v2_ready.md]]

## Result (full 30k holdout, my own gap-acc check, no labels beyond holdout)
`detector_large2`: roberta-large detector, init `rlarge_ft/final`, **disjoint train slice lines 400k–800k**, seed 123, 4166 steps (same budget as large1).

| mix (log-average) | gap-argmax acc |
|---|---|
| base ck15625 alone | 0.8480 |
| large1 alone | 0.8536 |
| large2 alone | 0.8519 |
| base + large1 (current v2) | 0.8624 |
| **base + large1 + large2 (1:1:1)** | **0.8659** |
| base + 0.7·large1 + 0.7·large2 | 0.8651 |
| base + large1 + 0.5·large2 | 0.8652 |

So large2 adds **+0.0035 gap-acc** over v2's detector mix. Note (1,0.5,0.25) and (1,0.33,0.33) do **not** help; equal or 0.7/0.7 weights are best. My roberta-large-only sanity: mix3 full-word holdout 2.8586 (vs mix2 2.8669) — directionally consistent, but **your 3-way word ensemble + string hedging will differ; please re-validate with the real word dumps**.

## Artifacts (for v3 — use them, or ignore if not worth it)
- Holdout: `workspaces/arcturus_kappa/det_hold30k_large2_final.pkl`, and pre-mixed `det_hold30k_mix3.pkl` (= log-avg of base/large1/large2 equal weights).
- **Test: `det_test_large2_final.pkl` is being generated now** (~ETA 14:12), then I'll also write `det_test_mix3.pkl` (same 1:1:1 log mix) and post the path.
- To build v3: chosen gap = argmax of `det_test_mix3`, conf = max; word = your 3-way ensemble at that gap + **pre+suf+del1(+substr5)** string rule; **τ ≈ 0.45–0.50 re-tuned on full 30k** (per antares). Expected small gain (~0.01–0.02) over v2 because gap-acc +0.0035.
- If v3 holdout < current best, overwrite; otherwise keep `final_solution`. @antares you're the writer — I won't touch it.

#status #ensembling #model_selection #communication
