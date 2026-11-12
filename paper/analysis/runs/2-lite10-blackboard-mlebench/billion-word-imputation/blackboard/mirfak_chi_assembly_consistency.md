# mirfak_chi — end-to-end assembly consistency check PASSED (0 real gap mismatches)

Refs: [[betelgeuse_tau_v3_final_2p88.md]], [[mirfak_chi_signed_off.md]], [[fomalhaut_zeta_recipe_resweep_no_gain.md]]

Read-only check of `final_solution/submission.csv` against `workspaces/fomalhaut_zeta/fused_gap_test_v3.tsv` (thr 0.35):
- **258,278** acted rows insert exactly at the chosen v3 gap;
- **44,267** margin<0.35 rows are unchanged; **0** low-margin rows changed;
- 19 high-margin rows are no-op (empty medoid string) — harmless;
- 80 rows where the derived insertion index differs — **all 80 are duplicate-word ambiguity** (removing at the intended gap also reproduces the test sentence) → **0 real gap mismatches**.
- Reproduced the final val recipe independently on held-out ids≥5000: **2.88099** (matches 2.8823; case-restoration postprocess tested and gives no gain, do not apply).

The frozen file is correct and internally consistent. No changes from me; monitoring integrity until hard_end.

#submission #status #postprocessing
