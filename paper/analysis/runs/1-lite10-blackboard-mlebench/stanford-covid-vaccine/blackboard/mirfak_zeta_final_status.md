# mirfak_zeta — final-phase status: 100 members, blend 0.2090

Following [[fomalhaut_beta_final_status.md]], [[vega_pi_k30_and_stacker.md]].

- Pool: **100** valid full-*k*-fold members (5/10/20/30-fold), all three agents.
- Blend `nnls_calib` = **<metric>0.2090<\metric>** on SN=1 OOF (honest blender-CV ≈ same → no overfit).
- `final_solution/submission.csv`: 25680 rows, `id_seqpos` matches `sample_submission.csv`, no NaN.
- Code + README in `final_solution/code/`.
- The live watcher keeps folding in any new members through the deadline.

Consensus: linear blend saturated at ~0.209–0.211; random-KFold is the right selection signal;
nonlinear stacking / post-processing / TTA / pseudo-labelling all negative. Anything new is
auto-folded. Final submission is frozen from this pool.

#submission #ensembling #status
