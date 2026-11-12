# spica_psi — v3 READY: mix3(arith) detector + kind2 strings → holdout <metric>2.7618</metric>

Re [[arcturus_kappa_mix3_arith_correction.md]] [[antares_upsilon_v3_plan.md]] [[spica_psi_v2_extended_strings.md]]

## Full-30k holdout (clean, exact char-Lev, baseline 5.5465), arithmetic mix3 `(base ck15625 + large1 + large2)/3`
Gap-acc mix3 arith = **0.8656**. Word = 3-way ensemble at chosen gap, strings = **prefix ∪ suffix ∪ single-char-deletion ∪ common**, expected-Lev + λ·len.
| τ | level | λ | metric | apply | loc |
|---|---|---|---|---|---|
| 0.45 | 1.0 | 0.05 | 2.7718 | 90.0% | 0.923 |
| **0.50** | **1.0** | **0.05** | **<metric>2.7618</metric>** | **87.7%** | **0.933** |
| 0.55 | 1.0 | 0.05 | 2.7690 | 85.4% | 0.943 |
(vs mix2 2.7824, mix2+prefix-only 2.7952, full-words 2.8248.) τ=0.50 stays optimal.

## Artifacts
- **`workspaces/spica_psi/decisions_v3.tsv`** — 265,447 rows (87.71% apply), ids 0..302643. **`submission_v3.csv`** — 302,644 rows, validated (0 bad, bounds/quoting OK, unchanged byte-identical).
- Reusable for @antares' own string rule (substr5 etc.):
  - **test word dumps at mix3 chosen gaps**: `workspaces/spica_psi/test_v3_{zs,ft,base}.pkl` (302,644; key = chosen gap)
  - **holdout word dumps at mix3 chosen gaps**: `workspaces/spica_psi/hold_mix3_{zs,ft,base}.pkl` (30,000)
  - gaps/conf from arcturus: `test_chosen_gaps_mix3.pkl` / `test_conf_mix3.npy`; `hold_chosen_gaps_mix3.pkl` / `hold_conf_mix3.npy`
  - my scorer: `eval_prefix3.py --kind 2 --det .../det_hold_mix3_arith.pkl --zs hold_mix3_zs.pkl --ft hold_mix3_ft.pkl --base hold_mix3_base.pkl`
- Word models same as before: roberta-large (0.5) ⊕ rlarge_ft (0.3) ⊕ base ckpt4000 (0.2).

@antares: please reproduce 2.7618, then either write `decisions_v3.tsv` or apply your **substring** rule to `hold_mix3_*`/`test_v3_*` (arcturus saw substr add ~0.004; if your full-30k number beats 2.7618 use it). Overwrite `final_solution` only with the better validated one. @arcturus: mix3 arith confirmed (I used your arith files, not log).

#insight #ensembling #postprocessing #submission #status #communication
