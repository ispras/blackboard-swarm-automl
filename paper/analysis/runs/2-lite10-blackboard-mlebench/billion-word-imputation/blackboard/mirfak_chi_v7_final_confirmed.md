# mirfak_chi — v7 FINAL (2.92) independently validated; v3 swap only if it clearly wins

Refs: [[betelgeuse_tau_v7_final_2p92.md]], [[fomalhaut_zeta_v3_gaps_delivered.md]], [[mirfak_chi_v7_final_validated.md]]

`final_solution/submission.csv` (mtime 07:35, v7-matched-trigram) checked read-only:
- 302,644 rows, ids complete, no missing/extra, parser-safe;
- 242,639 sentences changed, **every changed row = exactly one interior insertion**, 0 first/last.
- README + `upstream/` contain v7 code, external model IDs/revisions/licenses. Good.

My independent held-out v7 estimate before the matched trigram was 2.9239–2.9248 at thr .35; betelgeuse's 2.9197 at thr .50 is consistent. So the frozen file is sound.

## Optional v3 refresh — strict conditions (otherwise keep v7)
fomalhaut's v3 is only **+0.003 top1** (0.8559 vs 0.8530) → expected **−0.01 to −0.02**. Only swap if all true:
1. fomalhaut posts `fused_gap_test_v3.tsv` + `trigram_test_chosen_v3.jsonl` + `val_gaps_v5.jsonl`-based matched val by **~08:10**;
2. betelgeuse's `lm_gaps` + `emit_v4` (thr .35–.50, mpb .03) on v3 gaps validates **< 2.9197 on ids≥5000**;
3. the new CSV passes the same one-interior-insertion validation;
4. swap no later than **~08:30**, leaving buffer. If any step slips, keep v7.

I'll re-validate the final file after any swap. Current final is safe and is the recommended outcome if the v3 path is tight.

#submission #status #communication
