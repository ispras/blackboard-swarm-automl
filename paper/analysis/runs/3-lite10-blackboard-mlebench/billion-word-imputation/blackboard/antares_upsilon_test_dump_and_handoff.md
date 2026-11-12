# antares_upsilon — confirmations + test trigram dump + submission handoff

Re [[arcturus_kappa_fusion_3way_result.md]] [[spica_psi_mlm_ownership.md]] [[spica_psi_fusion_prototype.md]]

## Confirmations (your questions)
- `data/test_v2.txt` (my unzipped `workspaces/antares_upsilon/test_v2.txt`) = CSV header `id,sentence`, **302,644 data rows, ids 0..302643**, exactly the graded input. ✅
- `final_solution/` **does not exist yet / is empty**. Nobody has written it. I remain the writer; I won't create it until we agree on the final artifact.
- Quoting: use `csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)` (or QUOTE_MINIMAL) with `lineterminator='\n'`; writes `id,"sentence"` and escapes inner `"` as `""`. ✅ I will validate row count/ids/parse round-trip.

## Test trigram dump (per-gap gain) — IMPORTANT, already computing
`workspaces/antares_upsilon/ngram/test_gapcand.tsv.gz` (finishing now, ~302k rows in ~45 min):
`id<TAB>gap_k<TAB>word<TAB>gain<TAB>logscore`, **top-2 words per gap**, gap order, each gap's rows sorted by gain desc. `gap_k` = insert before token k (0-based), k∈[1, m-1] — same as your convention. Per-gap best = first row per `(id,gap_k)`.
I will also write the pkl equivalent matching spica's format:
`workspaces/antares_upsilon/ngram/tri_test_allgaps.pkl` → `list[dict{int gap_k: (str word, float gain, float logscore)}]`, order = test row order (ids 0..302643).
Holdout equivalent: spica already produced `workspaces/spica_psi/tri_holdout_allgaps.pkl` from my model; I can regenerate to my format if useful.

## Trigram-only fallback (independent, for reference / de-risking)
I added a **two-stage word refinement**: after picking the gap, re-rank `top-1000` successors(a) ∪ `top-1000` pred(b) (true-word candidate recall 91% vs 14% with the truncated set) → word exact 9.4%→22%, and choosing between top-2 gaps by refined gain:
**trigram-only holdout metric = <metric>4.4455</metric>** (thr 2.5, apply 45.9%). So even if neural pieces have issues, we have a solid ~4.45 submission available. Code `trigram_model.py` (`refine_word`), picks `ngram/refined_topk2_cap1000_30000.pkl`.

## Handoff I can execute on request
Send me either
1. a **decision file** `id<TAB>gap_k<TAB>word` (only where you insert; missing = leave unchanged), or
2. a ready `submission.csv` for review,
and I will: verify 302,644 unique ids, validate gap bounds, write `final_solution/submission.csv` + solution code + README (model sources: detector/MLM/spica checkpoints, HF model ID+revision+license, my n-gram). I will not insert anything not in the agreed decision file.

@spica_psi if your final MLM dump needs **candidate gaps**, arcturus will send those; my test per-gap gain dump is also a candidate source. Post the checkpoint HF model id/revision/license (e.g. `roberta-base`) so I can log sources.

#communication #plan #submission #status #inference #ensembling
