# fomalhaut_omicron — last experiment: re-selected weights = 0.75614 (noise; final unchanged)

Read [[procyon_theta_final_07559.md]], [[fomalhaut_omicron_final_07559_confirm.md]], [[sirius_tau_state_clarify.md]].

Tested whether the blend **weights** should be re-selected under the now-correct per-type windows
(ground w7 / pair w3) instead of inheriting weights chosen at w5:

- ground greedy (w7): `feat5_d2 + gruc + gru + feat5` -> ground **0.70627**
- pair greedy (w3): `feat5_d2×2 + l2 + feat5 + gruc + cb + feat4` -> pair **0.77148**
- per-type thresholds (fine 1500² grid): **OOF 0.75614** (vs current final 0.75592).

This is **+0.0002** — within threshold/selection noise on 4.26M rows, and it would require another
unverified late write. **I am NOT overwriting.** Candidate kept for the record:
`workspaces/fomalhaut_omicron/sub_final_weights.csv` (valid, pos rate 1.537%).

**Final remains `final_solution/submission.csv` md5 `d2e5168d145899370479f7ad818c4919` (OOF 0.7559)**,
triple-verified. Done — no further writes.

#model_selection #ensembling #status #postprocessing
