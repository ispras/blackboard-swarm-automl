# arcturus_kappa — status: GPU full (72GB) → holding inference; waiting on clean tri + rlarge_ft

Re [[arcturus_kappa_pipeline_2_95_and_word_model.md]] [[antares_upsilon_holdout_contamination.md]] [[spica_psi_mlm_ownership.md]]

## Confirmed 30k holdout (clean det/MLM features, **contaminated tri**) — baseline 5.5465
`ckpt_4000` detector + trigram + base-MLM `maxp` fusion, `roberta-large` words: **<metric>3.0488</metric>** @τ=0.60 (apply 82.4%, loc 0.928, word 0.599). First-5000 subset was 2.946 → 30k is the number to trust. Expect ~2.9 with final detector and ~2.8 if `rlarge_ft` words beat zero-shot.

## Agreed/noted
- **@antares_upsilon**: the **holdout trigram contamination is real and important** — my fusion weights + τ were fit on the contaminated holdout so they are NOT trustworthy for test (test tri_gain median 8.39 vs 1.70). I will **refit on `tri_holdout_clean_allgaps.pkl`** as soon as you publish it, and use your `tri_test_allgaps.pkl` for test. This is now the critical path for my decision file. Thank you for catching it.
- **@spica_psi**: saw you started **`roberta-large` MLM fine-tune** (`rlarge_ft`, 700k lines, ETA ~10:45) — excellent, that's the highest-value word job. Please (a) keep the final checkpoint, (b) ping me when `rlarge_ft` is ready and I'll A/B it vs zero-shot `roberta-large` on the holdout and use the winner for the test words. If time allows, 700k×~2 epochs > 1 epoch.
- **GPU**: currently **72GB/80GB used** (rlarge FT ~57GB + my detector ~14GB). I am **not** launching anything GPU-heavy now. My plan after ~10:45 (rlarge FT frees ~57GB):
  1. holdout: `rlarge_ft` vs zero-shot word A/B,
  2. **test compact detector pass** (final ckpt, ~30–40 min, fp16/capped),
  3. refit fusion on **clean** holdout → apply to test → `roberta-large(-ft)` words at chosen gaps,
  4. send `decisions.tsv` (`id<TAB>gap_k<TAB>word`) to antares.
- **Fallback already safe**: antares trigram-only ~4.45. If anything breaks, we still beat baseline 5.55.

## Requests
- @antares: please post when `tri_holdout_clean_allgaps.pkl` AND `tri_test_allgaps.pkl` are both ready (and their file paths/format). Also: is your **test** n-gram model built excluding nothing (test never in train) — confirm no other contamination.
- @spica: publish `rlarge_ft` path + exact word inference recipe (mask token insertion identical to zero-shot). Do **not** spend GPU on base-MLM test dumps — roberta-large(-ft) words dominate.

#communication #status #plan #ensembling #model_selection
