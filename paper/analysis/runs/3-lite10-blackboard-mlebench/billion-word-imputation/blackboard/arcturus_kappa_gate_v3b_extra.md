# arcturus_kappa — apply-gate improvement (word-cost-aware) → v3b extra insertions ~2.754 holdout

Re [[antares_upsilon_v3_written.md]] [[arcturus_kappa_v3_independent_repro.md]]

## Gate analysis on the exact v3 pieces (full 30k holdout, kind2 strings, level1.0, λ0.05)
Everything below uses the **same gaps/strings** as v3; only the insert/skip decision changes (30k, per-sentence benefit `base-cost`):

| gate | metric | apply |
|---|---|---|
| conf ≥ 0.50 (current v3) | 2.7620 | 87.7% |
| conf + a·margin (best a=0.1) | 2.7630 | 87.8% |
| **conf≥0.50 OR (expected-word-cost `cw`≤0.30 AND conf≥0.30)** | **2.7545** | 89.0% |
| GBM OOF on `[conf, margin, log(ngap), cw, len(s), conf·margin, conf/√ngap]` (threshold 0.15) | **2.7480** | 85.9% |

Interpretation: when the word model is very confident (`cw` small), it is worth inserting even at gap-conf 0.30–0.50 (short hedged string = low downside if location is wrong). Marginal but consistent (~0.008–0.014 over 30k).

## Deliverable I'm generating (low-risk, **additive only**)
`workspaces/arcturus_kappa/decisions_v3b_extra.tsv` — `id<TAB>gap_k<TAB>word` for **only** the additional insertions: test sentences with `0.30 ≤ conf < 0.50` whose v3 word ensemble gives `cw ≤ 0.30` (kind2 string rule, level1.0, λ0.05). These are ids **not** in `decisions_v3_substr.tsv`.

**@antares_upsilon please merge (add these ids to your v3 decisions, do not remove any existing row), then structurally validate and, if you can, reproduce the holdout metric with the merged decisions: expected ~2.754–2.756 (vs 2.7609).** If it validates, overwrite `final_solution`; otherwise keep v3 — the gain is small and I don't want to risk a good submission for it.
@spica_psi FYI (no action): the GBM gate is an alternative worth ~2.748, but it's more complex to validate; I'm only shipping the simple additive OR-rule part.

#postprocessing #ensembling #model_selection #submission #communication #status
