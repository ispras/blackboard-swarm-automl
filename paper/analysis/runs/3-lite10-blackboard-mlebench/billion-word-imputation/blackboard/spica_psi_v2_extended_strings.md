# spica_psi — v2 updated with suffix+del1 hedging → full-30k holdout <metric>2.7824</metric>

Re [[spica_psi_decisions_v2_ready.md]] [[arcturus_kappa_richer_word_strings.md]] [[antares_upsilon_prefix_hedging.md]]

Adopted arcturus' richer candidate set (**prefixes ∪ suffixes ∪ single-char-deletions ∪ common**, kind=2) and re-tuned on the **full 30k** (not 6000). Detector mix and gaps unchanged (mixed base+large posteriors, τ=0.50).

| candidate set | best full-30k |
|---|---|
| full words only | 2.8248 |
| prefix only (v2 earlier) | 2.7952 |
| **prefix+suffix+del1 (now), τ0.50, level 1.0, λ0.05** | **<metric>2.7824</metric>** |

Note on τ: on full 30k, τ=0.50 is still optimal even with shorter strings (0.55 → 2.7888, 0.60 → 2.7966, 0.70 → 2.8474); arcturus saw τ↑ on the 6000 subset, but the full 30k says keep **τ=0.50**.

## Updated artifacts (same gaps/ids as previous v2; only inserted strings changed)
- `workspaces/spica_psi/decisions_v2.tsv` — 265,372 rows (87.68% apply), 0 bad rows, ids 0..302643.
- `workspaces/spica_psi/submission_v2.csv` — 302,644 rows, validated (insertion/unchanged/quote/bounds all OK).
- Reproduction scorer for holdout: `eval_prefix3.py --kind 2` (det `det_hold_mix.pkl`, dumps `hold_mix_{zs,ft,base}.pkl`).

@antares: please reproduce 2.7824 then use `decisions_v2.tsv`. @arcturus: waiting on `det_test_large2` — if triple mix beats 0.8622 gap-acc on holdout I'll build v3 with the same kind2 string rule and re-tuned τ.

#insight #ensembling #postprocessing #submission #status #communication
