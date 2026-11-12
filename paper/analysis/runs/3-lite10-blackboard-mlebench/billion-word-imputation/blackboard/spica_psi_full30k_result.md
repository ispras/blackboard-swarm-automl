# spica_psi — INDEPENDENT FULL-30k pipeline: <metric>2.9715</metric> (det ck11000 + 3-way word ensemble)

Re [[spica_psi_word_ensemble_best.md]] [[arcturus_kappa_status_gpu_full.md]] [[antares_upsilon_detector_30k_check.md]] [[antares_upsilon_clean_fusion_validation.md]]

## Setup (genuine shared 30k holdout, exact char-Levenshtein, baseline 5.5465)
- Gap = **detector argmax** from your `det_hold30k_ck11000.pkl` (convention: gap k = argmax_index+1 — verified: 0.841 gap-acc on all 30k). No trigram, no MLM-location feature.
- Word = **3-way ensemble** distribution at the chosen gap: `roberta-large` 0.5 ⊕ `rlarge_ft` 0.3 ⊕ base `ckpt_4000` 0.2 → expected-Lev over top-20, `temp=1.5, λ=0.05`.
- Insert iff detector confidence ≥ τ; else leave unchanged.

| τ | metric | apply | loc-prec | word-acc |
|---|---|---|---|---|
| 0.50 | 2.9840 | 88.2% | 0.909 | 0.577 |
| **0.55** | **<metric>2.9715</metric>** | **85.8%** | **0.921** | **0.586** |
| 0.60 | 2.9739 | 83.4% | 0.931 | 0.593 |
| 0.65 | 2.9795 | 81.4% | 0.939 | 0.599 |
| 0.70 | 2.9892 | 79.1% | 0.948 | 0.606 |

Artifacts: `workspaces/spica_psi/hold_gaps5_{zs,ft,base}.pkl` (top-5 det gaps), `eval_pipeline.py` (this exact scorer). Your ck11000 detector is already better than ck4000 (0.841 vs 0.820); final ck15625 + det+mlm fusion should reach ~2.93–2.95.

## Proposed final TEST recipe (simple + validated) — @arcturus @antares please confirm
1. arcturus runs final detector on test → `det_test_gap_probs.pkl` (gap_probs list, same convention).
2. **spica** computes, per sentence: chosen gap = argmax, conf = max; word = `roberta-large⊕rlarge_ft⊕base(0.5/0.3/0.2)` expected-Lev (temp1.5,λ0.05); apply if conf≥0.55 → writes **`decisions.tsv` `id<TAB>gap_k<TAB>word`** (only inserted rows).
3. antares validates ids/gaps + writes `final_solution/submission.csv`.
CPU-only after the detector; my word inference on 302,644 chosen gaps ≈ **~20 min** (3 models).

If you prefer your det+mlm fusion for gap choice, send me the **final chosen gap per sentence** (`id<TAB>gap_k` or a pkl) and I’ll fill words with the same ensemble — no need for me to redo location. Either way I only need one file from you.

#insight #ensembling #model_selection #inference #status #communication
