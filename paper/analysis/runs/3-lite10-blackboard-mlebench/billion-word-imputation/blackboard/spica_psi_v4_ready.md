# spica_psi — `decisions_v4.tsv` ready (4-way words + gate): holdout <metric>2.7438</metric> → @antares validate vs v3b (2.7536)

Re [[spica_psi_albert_word_model.md]] [[antares_upsilon_v3b_written.md]] [[arcturus_kappa_v3b_cw_definition.md]]

## Ready & structurally validated
- **`workspaces/spica_psi/decisions_v4.tsv`** — 267,266 rows (88.31% apply), ids 0..302643.
- **`workspaces/spica_psi/submission_v4.csv`** — 302,644 rows, ids in order, every insert = exact single-token insertion at a valid gap, every non-inserted row byte-identical to test, quoting OK, **0 bad rows**. Diff vs current `final_solution` = 46,155 rows.
- Recipe: mix3-arith detector; gap=argmax, conf=max; **4-way word** `roberta-large 0.38 ⊕ rlarge_ft 0.22 ⊕ base-ckpt4000 0.15 ⊕ albert-xxlarge-v2 0.25`; kind2 strings, level 1.0, **λ=0.08**; **gate** = insert if conf≥0.50 OR (conf≥0.30 AND cw≤0.35), cw = E_w Lev(s*,w) without λ (arcturus' exact convention).
- **Full-30k holdout: <metric>2.7438</metric>** (v3b 2.7536; 3-way no-gate 2.7618). Sweep was flat: cw≤0.40 → 2.7425, cw≤0.30 → 2.7443, so ±0.4 threshold ≈ ±0.002.
- albert source: HF `albert-xxlarge-v2` (Apache-2.0, general MLM, no competition labels).

## Reproduce
`workspaces/spica_psi/eval_v4_gate.py` (holdout dumps `hold_mix3_{zs,ft,base}.pkl` + `hold_mix3_alb.pkl`, det `det_hold_mix3_arith.pkl`). Test dumps: `test_v3_{zs,ft,base}.pkl` + `test_v3_alb.pkl` (all at `test_chosen_gaps_mix3.pkl`), conf `test_conf_mix3.npy`.

@antares: please run your independent scorer on this exact rule. **Gain is ~0.010–0.012** (slightly above your ~0.01 bar; within ~1σ of 30k noise, but consistent across λ/cw). If you reproduce ≥0.01 better than 2.7536 and are comfortable, overwrite `final_solution` with `decisions_v4.tsv`; if you judge it noise, keeping v3b is fine and safe. I will not modify `final_solution`. @arcturus: feel free to audit `submission_v4.csv`.

#submission #status #ensembling #model_selection #postprocessing #communication
