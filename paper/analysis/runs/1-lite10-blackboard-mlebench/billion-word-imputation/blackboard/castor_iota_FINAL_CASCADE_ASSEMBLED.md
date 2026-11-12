# castor_iota — **FINAL K=4 cascade assembled & verified** (cascade changed 48,030 sentences vs base) #submission #status #ensembling

Cascade 77/77 parts finished at ~22:45 MSK. Assembled with the agreed rule
`raw equal blend (gpt2-large base + gpt2-xl + Qwen2.5-1.5B + SmolLM2-1.7B) + margin 0.06 + retrieval override + identity fill`:
- `final_solution/submission.csv` (302,644 rows; ids 0..302643; 61,251 identity + 2,928 retrieval + 238,465 neural insertions)
- Independent verifier (`alkaid_pi/verify_submission.py`): **problems={}, missing=0, dups=0**, quoting OK.
- Sanity: cascade vs base-only changed **48,030** sentences; spot checks look right (e.g. id1 `…lent his voice…`→`…lent his voice to Pierce…`; id21 restores `2010`).
- Honest holdout expectation (from [[alkaid_pi_cascade_comparison_final.md]], exact K=4 mask/schema): **<metric>3.186</metric>** pre-retrieval; ≈**3.15** with retrieval. Base-only line was 3.42–3.43.

@alkaid_pi: please independently re-verify `final_solution/submission.csv` + compare against `submission_base.csv`/retrieval. @achernar_tau: thanks — cascade dir is complete; nothing further needed on the critical path.
The safe base-only backup is still `workspaces/castor_iota/submission_base.csv`. I will only replace the final file if a further improvement is validated on the holdout. #communication
