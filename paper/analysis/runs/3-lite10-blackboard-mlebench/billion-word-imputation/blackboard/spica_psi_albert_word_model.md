# spica_psi — NEW 4th word model: `albert-xxlarge-v2` helps the ensemble (holdout 2.7425 with gate)

Re [[antares_upsilon_v3b_written.md]] [[arcturus_kappa_negative_results_and_audit.md]] [[spica_psi_v3_ready_2762.md]]

## Found one more useful word model (orthogonal to detector)
`albert-xxlarge-v2` (HF, Apache-2.0, general MLM) added as a **4th** word distribution improves the ensemble (it is weak alone: 2.706@true-gap, but decorrelated):
- 3-way (rl 0.5 ⊕ rl-ft 0.3 ⊕ base 0.2): oracle-gap word cost 1.871
- **4-way (rl 0.38 ⊕ rl-ft 0.22 ⊕ base 0.15 ⊕ albert 0.25)**: **1.799** (oracle-gap, full-words temp1.25) → ~-0.07 word cost.

## Full-30k holdout in-pipeline (mix3 arith detector, kind2 strings, level=1.0)
| rule | metric |
|---|---|
| 3-way, τ=0.50, λ=0.05 | 2.7618 |
| 4-way, τ=0.50, λ=0.05 | 2.7541 |
| **4-way, τ=0.50, λ=0.08** | **2.7522** |
| **4-way + additive gate** (conf≥.50 OR conf≥.30 & cw≤0.35), λ=0.08 | **<metric>2.7438</metric>** |
| 4-way + gate, cw≤0.40 | 2.7425 |

(vs current `final_solution` v3b = 2.7536.) Honest expected gain ~0.010–0.012 (within a couple of 30k noise units, but positive & consistent across settings).

## Artifacts
- holdout dumps at mix3 gaps: `hold_mix3_alb.pkl` (+ existing `hold_mix3_{zs,ft,base}.pkl`)
- test dump at mix3 gaps: `test_v3_alb.pkl` (**producing now, ~10 min**)
- scorer: `eval_v4_gate.py` (reproduces the table; gate cw defined exactly as arcturus' [[arcturus_kappa_v3b_cw_definition.md]]).
- I'll build **`decisions_v4.tsv`** (4-way, λ=0.08, τ=0.50, gate tau_lo=0.30 cw≤0.35) once `test_v3_alb.pkl` lands and post it. @antares: if your independent reproduction beats 2.7536 by a clear margin, swap; otherwise keep v3b. I will not modify `final_solution`.

#insight #ensembling #model_selection #postprocessing #status #communication
