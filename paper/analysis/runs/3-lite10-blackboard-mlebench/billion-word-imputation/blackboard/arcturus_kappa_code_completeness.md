# arcturus_kappa — two small pre-freeze notes (@antares): code completeness + weight sweep already done

Re [[antares_upsilon_v3b_adopted.md]] [[arcturus_kappa_word_model_exhausted.md]]

1. **`final_solution/decisions.tsv` verified byte-identical to my `decisions_v3b_merged.tsv`** (269,536 rows, diff empty). `DECISION.md` accurately documents v3b including the additive gate. ✅
2. **Code completeness (minor):** for full reproducibility of the v3b gate, please add these two scripts from my workspace to `final_solution/code/` (they are the only v3b-specific pieces not already there):
   - `workspaces/arcturus_kappa/scripts/make_v3b_extra.py` (the `0.30≤conf<0.50 & cw≤0.30` additive rule + string choice),
   - `workspaces/arcturus_kappa/scripts/build_gate_cache.py` (feature/`cw` computation used to calibrate it).
   Not strictly necessary (DECISION.md describes the rule), just nice to have.
3. **Weight/λ sensitivity — already answered**, don't spend more time: with the deployed dumps and the actual metric (full 30k, kind2 strings, level1.0, λ0.05, τ0.5), the current `0.5/0.3/0.2` is best: `2.762` vs `0.6/0.3/0.1=2.772`, `0.4/0.4/0.2=2.769`, `0.5/0.25/0.25=2.766`, `0.5/0.5/0=2.773`, `1/0/0=2.825`, `0/1/0=2.852`. So no weight change.

I'll do the **final audit of the frozen `final_solution/submission.csv`** once you post freeze (and again near 18:55): row count/ids, header/quoting, single-token insertion validity, no control chars, and that `decisions.tsv` matches the submission. Ping me when frozen.

#communication #submission #status
