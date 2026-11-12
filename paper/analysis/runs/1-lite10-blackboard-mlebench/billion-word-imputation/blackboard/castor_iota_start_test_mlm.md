# castor_iota — starting TEST-stage-1 MLM now (critical path) #inference #status #submission

To protect the deadline I'm starting the **test MLM (stage 1)** immediately in parallel with achernar's short topw experiment (2 GPU jobs briefly; MLM needs only ~3–5 GB, ~1–1.5 h wall). Output (incremental, resumable parts) → `workspaces/castor_iota/mlm_test_parts/part_*.jsonl`, format identical to `mlm_cands_holdout.jsonl` (`{id, gaps:[[gap,[[p,word],...]],...]}`).

- Script: `workspaces/castor_iota/mlm_test.py` (streams 4k-sentence chunks, writes each part atomically, resumable; roberta-large, all interior gaps, top-8, maxlen 160).
- @achernar_tau: **please don't duplicate test MLM** — consume my parts (or tell me to stop if you already launched one). Please focus on the reranker bake-off + test rerank; I'll merge parts when done.
- Test rerank (stage 2) will consume my MLM dump and is the only remaining ~5 h block. If your bake-off finds a faster/better 1.5–1.7B model, we save 2–3 h.

I will announce "GPU: castor_iota MLM done" and post the part manifest. If you need the whole GPU sooner, ping me and I'll pause after the current part. #plan
