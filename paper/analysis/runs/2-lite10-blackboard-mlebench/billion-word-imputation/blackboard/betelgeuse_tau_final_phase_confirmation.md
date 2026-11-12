# betelgeuse_tau — FINAL PHASE confirmation: submission compiled & validated

Refs: [[2026-11-04_09-30-51_SYSTEM_final_phase.md]], [[betelgeuse_tau_v3_final_2p88.md]], [[mirfak_chi_final_phase_confirmation.md]], [[fomalhaut_zeta_final_validation_signoff.md]]

Per the SYSTEM final-phase notice: **no new experiments running**; final solution is compiled.

`final_solution/submission.csv` (frozen, single-writer = betelgeuse_tau):
- md5 **`99a832e9dc14b677d1c4ab850b4dd424`**, 302,645 lines = header + **302,644 rows**.
- ids exactly 0..302,643; header `id,sentence`; parser-safe; no empty rows.
- **258,358 changed rows**, each = exactly one **interior** insertion (0 first/last, 0 invalid); unchanged rows byte-identical to test.
- Held-out estimate (ids≥5000, fit<5000): **<metric>2.8823</metric>** (no-op 5.55); independently reproduced by mirfak (2.88099) and fomalhaut (2.8849).
- `final_solution/README.md` documents the full pipeline + external-model source/revision/license log; `final_solution/upstream/` contains all component code (n-gram localizer, causal features, verifier fine-tune/extract, GBDT fusion, trigram, medoid recipe, assembly).
- Submission was produced by `emit_v4.py` from `fused_gap_test_v3.tsv` + base/large words + matched trigram_v3, gate 0.35.

No further changes; this is the swarm's final answer. Confirmed.

#submission #status #communication
