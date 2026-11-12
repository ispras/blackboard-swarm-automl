# mirfak_chi — FINAL QA (last check): PASSED, frozen at md5 `99a832e9…`, val ≈2.88

Refs: [[betelgeuse_tau_v3_final_2p88.md]], [[fomalhaut_zeta_final_validation_signoff.md]], [[fomalhaut_zeta_recipe_resweep_no_gain.md]], [[mirfak_chi_final_v3_qa.md]]

Last read-only integrity check of `final_solution/submission.csv` (unchanged, md5 `99a832e9dc14b677d1c4ab850b4dd424`):
- header `id,sentence`; **302,644 rows**; ids 0..302,643 complete; 0 missing / 0 extra; parser-safe.
- **258,358 changed rows**, each = exactly one **interior** insertion (removal reproduces the test sentence); **0 first/last** insertions.
- README + `upstream/` hold the v3/v5 code and the external-model source log (roberta-base, roberta-large, gpt2-medium; revisions + MIT).
- fomalhaut's final recipe re-sweep: no gain (2.8839/2.8842 vs frozen 2.8823) → freeze confirmed.

**Final: no-op 5.55 → 4.19 → 3.23 → 2.99 → 2.92 → <metric>2.8823<\metric>** (held-out ids≥5000). Key unlock: mask-hidden-state gap verifier (v1 frozen → v2 fine-tuned roberta-base → v3 fine-tuned roberta-large), fused with fomalhaut's 20M n-gram + causal GPT-2 + MLM, gated medoid words. No test labels used; all selection on a train holdout. Signing off — no further changes from me.

#submission #status #communication
