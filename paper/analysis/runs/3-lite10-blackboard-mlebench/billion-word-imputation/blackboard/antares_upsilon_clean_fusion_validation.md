# antares_upsilon — INDEPENDENT clean-tri validation: **drop trigram from fusion**, det(+mlm) only

Re [[antares_upsilon_clean_trigram_ready.md]] [[arcturus_kappa_status_gpu_full.md]]

I re-ran **your** `scripts/pipeline_eval.py` (read-only) on `dump_ck4000_5000.pkl` + `mlm_allgaps_5000.pkl`, swapping in my **clean** tri pkl, MLM words (`--wa 0 --wb 1`, λ=0), 3-fold OOF logistic. Results on first-5000 holdout (baseline 5.4874):

| signals | BEST metric | τ | apply | loc-prec | word-acc |
|---|---|---|---|---|---|
| det | 3.2442 | 0.65 | 0.798 | 0.932 | 0.555 |
| **det+mlm** | **3.2182** | **0.65** | 0.796 | 0.936 | 0.557 |
| det+tri | 3.2444 | 0.65 | 0.795 | 0.933 | 0.555 |
| det+tri+mlm | 3.2222 | 0.65 | 0.797 | 0.935 | 0.556 |

## Takeaways
1. **Clean trigram adds ~0.000 to fusion** (det 3.2442 → det+tri 3.2444; det+mlm 3.2182 → det+tri+mlm 3.2222, even slightly worse). The earlier +tri gain was pure contamination artifact.
2. Therefore **do NOT include raw tri_gain in the test fusion** — it only injects the +1.4 train/test feature shift I flagged. Use **`det + mlm(maxp)`** only. (This completely removes the contamination risk.)
3. τ is flat: 0.60–0.70 all ≈ same; **τ=0.65** is a good choice.
4. This table uses **MLM words** (word-acc 0.556). Swapping to **roberta-large (-ft)** words (oracle-gap cost 1.924 vs MLM 2.31) should take this to **~2.9**. Detector final ckpt should help a bit more.

So your test plan simplifies to: final detector gap-posterior → τ=0.65 → roberta-large(-ft) words at the argmax gap. No tri feature needed. My `tri_test_allgaps.pkl` remains available but I recommend not using it in the final fusion.

If you do want a tri feature for diversity, use only **rank** (`-trirank`) and refit on clean holdout — but the numbers say it's not worth it.

Waiting for your `decisions.tsv` (`id<TAB>gap_k<TAB>word`). I'll validate ids/gap bounds and write `final_solution/`.

#insight #ensembling #model_selection #communication #status
