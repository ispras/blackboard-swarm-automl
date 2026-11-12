# mirfak_chi — final submission validated (format + one-interior-insertion check)

Refs: [[betelgeuse_tau_final_frozen_2p9895.md]], [[fomalhaut_zeta_v2_final_file_ready.md]], [[mirfak_chi_conflict_dual_large_verifier.md]]

Independent validation of the current `final_solution/submission.csv` (read-only):
- header `id,sentence`, **302,644 rows**, ids exactly 0..302,643, no missing/extra;
- **250,778 sentences changed**, each differs from the test sentence by **exactly one inserted token**;
- parser-safe (`csv` reads 2 columns, `""` escaping intact);
- **0 insertions at first or last position** (all interior, as the task requires);
- unchanged 51,866 rows are byte-identical to the test sentence.

I also reproduced the recipe on the **held-out ids≥5000** (n=14,948): `0.3·base + 0.7·large + 0.3·trigram_v2`, medoid depth 3, mpb .03, gate .35 → **2.9880** (mpb .05 → 2.9905), consistent with the frozen 2.9895. Neighbour-duplicate suppression (9/14,948 rows) does **not** help. So the current final is sound; keep it unless the roberta-large verifier yields a validated held-out gain.

Large-verifier watch: two duplicate roberta-large fine-tunes are still running (PIDs 44765 fomalhaut / 44794 betelgeuse), GPU 68–79 GB. I've flagged the duplication in [[mirfak_chi_conflict_dual_large_verifier.md]]; recommend **one** extraction owner + one v7 refit. When a `verif_large_gaps_val.jsonl` appears I'll independently score standalone/fused top1 on ids≥5000 before any swap.

#submission #status #communication
