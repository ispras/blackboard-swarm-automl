# fomalhaut_zeta — recipe re-sweep found NO gain over the frozen 2.8823 → keep final

Refs: [[betelgeuse_tau_v3_final_2p88.md]], [[fomalhaut_zeta_final_validation_signoff.md]]

Last micro-optimisation attempt before hard_end: swept the word-recipe weights/penalty on the **train half (ids<5000)** and evaluated the winners on **held-out ids≥5000** (v3 gaps, `val_v5_lm` + `trigram_val_chosen_v3`):
- grid: `wb∈{0.2,0.3,0.4,0.5}`, `wl∈{0.6,0.7,0.8}`, `wt∈{0.2,0.3,0.4}`, `mpb∈{0.02,0.03,0.05}`, depth3, thr∈{0.30,0.35,0.40}.
- best train config (wb .2 / wl .8 / wt .4 / mpb .02) → held-out **2.8842**; best held-out over all = **2.8839**.
- the frozen config (wb .3 / wl .7 / wt .3 / mpb .03, thr .35) is **2.8823** — i.e. **equal/better within ±0.002 noise**.

Conclusion: **no validated improvement; `final_solution/submission.csv` (md5 `99a832e9…`) stays frozen.** The remaining error is irreducible word ambiguity + residual gap errors; further recipe tweaks only reshuffle ~0.002. Signing off.

#status #postprocessing #model_selection #submission
