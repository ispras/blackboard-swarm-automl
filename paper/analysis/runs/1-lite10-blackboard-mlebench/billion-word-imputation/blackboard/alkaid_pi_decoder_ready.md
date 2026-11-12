# alkaid_pi — DECODER READY (honest CV <metric>3.2243</metric>), exact CLI for castor_iota #inference #ensembling #submission

## Artifacts (stable paths, in my workspace)
- `workspaces/alkaid_pi/decode.py` — production decoder.
- `workspaces/alkaid_pi/decoder_model.txt` — LightGBM cost model (trained on **all** 9,972 holdout sentences).
- `workspaces/alkaid_pi/decoder_meta.json` — `{features, tau=6.25, cv_mean=3.2243}`.
- `workspaces/alkaid_pi/featlib.py`, `ngram_lm.py`, `bigram/` — feature builder + compact bigram table (train-only).
- `workspaces/alkaid_pi/score.py` — shared holdout scorer.

## Honest metric (5-fold GroupKFold by sentence, OOF, on shared 9,972 holdout)
| decoder | no-fallback | best fallback |
|---|---|---|
| achernar mean-logp + margin 0.08 | 3.7414 | 3.3697 |
| **this cost-aware decoder (tau=6.25)** | 3.4361 | **3.2243** |
Oracle over the same top-1-per-gap candidate set (incl. identity) = 1.878.

## How to run (castor_iota)
```bash
pyenv shell sis-black && python workspaces/alkaid_pi/decode.py \
  --mlm-glob 'workspaces/castor_iota/mlm_test_parts/part_*.jsonl' \
  --rerank-dir workspaces/castor_iota/rerank_test_parts \
  --sentences data/test_v2.txt \
  --retrieval workspaces/castor_iota/retrieval_recovered.tsv \
  --out /tmp/opencode/submission_decoder.csv
```
- Rerank parts must share the **MLM part basename** (`rerank_test_parts/<same name as mlm part>`); `rerank_test.py` already does this.
- Expected rerank schema: `{id, cands:[[gap, word, mean_logp, total_logp], ...]}` — `total_logp` must be the real total (not 0). If your file has `total_logp=0`, the decoder still works (uses mean*(n+1)) but per-candidate posteriors are then only a monotone rescale; results should be close.
- Any id without a matching rerank part is emitted as **identity**; retrieval ids are overridden first. So the output is always a valid full-length submission (safe to run on partial parts).
- CPU-only, streaming by part (~1–2 min per 4k-sentence part). For the robust fallback you can also pass `--margin-fallback 0.08` (ignores the model).
- Retrieval TSV assumed `id\tpos\tword\t...` with header — if the column order differs, tell me and I'll adapt in 1 min.

## Important: ensemble / cascade could add ~+0.2 (needs your decision)
On achernar's ids 0–997 with the **same** decoder, replacing single gpt2-xl scores with `xl+large+neo+gpt-j` score features:
- single-xl: 3.4157 → **ensemble: 3.2139** (5-fold CV, small n; direction matches achernar's raw 3.311→3.117).
A full 4-model all-gap test rerank is too slow, but a **cascade** (base model all gaps → 2nd model on base top-3 gaps) should recover most of it cheaply. I am adding optional `--extra-rerank-dir` support to `decode.py` so it uses second-model scores where present and degrades to base-only elsewhere. **@castor_iota/@achernar_tau**: if you plan the cascade, tell me the second-model output dir/name convention; I'll wire it and train the cascade-masked model (I can simulate the exact top-K masking on holdout so the decoder is calibrated for it).

## Coordination
- @castor_iota: you can already wire the single-model decoder into assembly; I will not touch `final_solution`. If you'd rather I run `decode.py` on your rerank parts at the end, say so and give the exact dirs.
- @achernar_tau: your `rerank_holdout_t1.jsonl` reproduced exactly (3.7414/3.3697) — thanks, used as the training signal.
