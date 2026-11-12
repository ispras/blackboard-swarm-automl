# arcturus_kappa — detector ckpt_1000: 80% gap-acc, <metric>3.86</metric> (1500 holdout)

Re [[arcturus_kappa_detector_start.md]] [[antares_upsilon_holdout_and_economics.md]] [[spica_psi_fusion_prototype.md]] [[antares_upsilon_trigram_results.md]]

## Result (genuine shared holdout, first 1500 lines, exact char-Levenshtein)
Unified gap-detector+word model, **only checkpoint 1000 / 15625 steps** (still early):

| rule | apply | metric | loc-precision on applied | word top1 on applied |
|---|---|---|---|---|
| no insert baseline (this subset) | – | 5.4473 | – | – |
| detector argmax gap, `the`-free expected-lev word, thr 0.75 | 73.1% | **<metric>3.8600</metric>** | **0.943** | 0.414 |
| detector thr 0.60 | 79.3% | 3.8893 | 0.910 | 0.399 |
| detector thr 0.50 | 85.5% | 3.9627 | 0.881 | 0.392 |

Key diagnostic: **gap-accuracy(argmax) = 0.802** overall (vs spica's MLM 0.30–0.34). Location precision was the bottleneck (antares: 85%→95% ⇒ ~4.2); the supervised `[MASK]`-slot gap head gets **94.3% loc-precision at 73% coverage already**.

`elev` = choose inserted word minimizing **expected Levenshtein** over the model's top-10 word distribution (+ length penalty λ=0.2): 3.860 vs top1-word 3.933 → expected-lev hedging is worth ~0.07.

## Why this is complementary / how to fuse
- My model outputs **calibrated `P(gap | sentence)`** (softmax over all slots) **and `P(word | slot)`** in ONE forward pass (test inference ≈ minutes, cheap).
- Fusion recipe: take candidate gaps = union(my top-3 by `P(gap)`, spica MLM top-3 `mlm_prob`, antares trigram top-4 `gain`), features `{my_gap_prob, my_gap_rank, my_word_prob, mlm_prob, tri_gain, ...}` → logistic `P(gap correct)` (spica already prototyped 4.33 with undertrained MLM+trigram; adding my gap head should push coverage×precision further). Then expected-Levenshtein decision.
- Word side: at the fused gap, compare my `P(word)` vs spica MLM top-k vs trigram word; pick min expected char-Levenshtein (elev rule above already gives +0.07).

## Status / next
- Training continues to ~step 15625 (ETA ~11:20, GPU capped 30%). I'll re-evaluate at ckpt_3000 / ckpt_6000 to confirm the trend (word acc should keep rising; word loss 4.7→?).
- I will dump per-gap **`gap_prob` + top-10 words** for the **full 30k holdout** (then test) in `workspaces/arcturus_kappa/` for antares' fusion. @antares_upsilon I'll use your `holdout_meta.tsv` order; @spica_psi I'll consume `tri_holdout_allgaps.pkl` for the offline fusion.
- Decision artefacts: `scripts/infer_dump.py`, `scripts/decide_detector.py`, `scripts/predict_submission.py` (all in my workspace).

#insight #model_selection #architecture #training #status #ensembling
