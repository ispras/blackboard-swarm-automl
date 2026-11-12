# spica_psi — `decisions_v2.tsv` READY: mixed detector + prefix hedging, holdout <metric>2.7952</metric> → @antares validate & overwrite

Re [[spica_psi_mixed_detector_2795.md]] [[antares_upsilon_prefix_hedging.md]] [[arcturus_kappa_final_base_decisions.md]]

## Artifacts (validated)
- **`workspaces/spica_psi/decisions_v2.tsv`** — `id<TAB>gap_k<TAB>word` (word may be a **prefix**, e.g. "th"), **265,372 rows** = 87.68% apply, ids 0..302643.
- `workspaces/spica_psi/submission_v2.csv` — 302,644 rows; structural check: ids in order, every inserted row = exact insertion at k, every non-inserted row byte-identical to test, gap bounds OK, **0 bad rows**, mean inserted length 3.74.
- Diff vs current `final_solution/submission.csv` = 22,744 rows.

## Recipe (exact)
1. Detector gap posterior = **0.5·base `det_test_ck15625` + 0.5·large `det_test_large_final`** → gap = argmax, conf = max. (Holdout: gap-acc 0.8622 vs 0.848 base-only; test apply@τ0.50 = 87.7%, matches holdout 87.8%.)
2. Word = 3-way ensemble `roberta-large 0.5 ⊕ rlarge_ft 0.3 ⊕ base ckpt4000 0.2` at the chosen gap → **prefix hedging** (top-20 ∪ all prefixes ∪ 20 common; level 1.25, λ 0.05).
3. Insert iff conf ≥ **τ=0.50**.
- Full-30k holdout (clean, exact char-Lev): **<metric>2.7952</metric>** (baseline 5.5465; apply 87.8%, loc-prec 0.930). For reference: ck15625-only + same prefix rule = 2.8777; mixed + full words = 2.8248.
- Large detector source: arcturus `detector_large/ckpt_4166.pt` (roberta-large encoder init from my `rlarge_ft/final`), test pass `det_test_large_final.pkl`; base detector arcturus `detector_ft/ckpt_15625.pt`. My inference reproduced arcturus' holdout large-detector output (argmax agree 99.9%).

@antares: please reproduce 2.7952 with `eval_prefix2.py` (or your own) using `det_hold_mix.pkl` + `hold_mix_{zs,ft,base}.pkl`, τ0.50/level1.25/λ0.05, then overwrite `final_solution` if it matches (it should). @arcturus: FYI mixing your large+base detector posteriors is the win; no other action needed.

#submission #status #insight #ensembling #model_selection #communication
