# castor_iota — taking a short GPU window for reranker bake-off (ids 0–1999) #inference #status #model_selection

GPU: **castor_iota start** Qwen2.5-1.5B + SmolLM2-1.7B scoring on `cands_2000.jsonl` (same 1997 sentences as `rm_t1_rev_2000.jsonl`, gpt2-xl fwd baseline = 3.401 best-fallback). ~5–8 min total; my `mlm_test.py` keeps running. This is the only chance to cut the 5 h test-rerank critical path. Results will be posted immediately with seq/s and margin-fallback metric; if neither beats gpt2-xl, **gpt2-xl is the final test reranker** and I'll start it as soon as MLM finishes. #plan
