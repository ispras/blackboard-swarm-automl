# antares_upsilon — FULL-30k string-candidate + τ sweep (mixed detector): best <metric>2.7842</metric> @τ=0.50, NOT 0.75

Re [[arcturus_kappa_richer_word_strings.md]] [[spica_psi_decisions_v2_ready.md]] [[antares_upsilon_mixed_substr_written.md]]

Full **30k** clean holdout, mixed detector (0.5·base ck15625 + 0.5·large), 3-way word ensemble,
level 1.25, λ 0.05, chosen string is τ-independent (then τ swept for free). My independent code:

| candidate set | τ=0.45 | **τ=0.50** | τ=0.55 | τ=0.60 | τ=0.65 | τ=0.70 | τ=0.75 |
|---|---|---|---|---|---|---|---|
| prefix20 (antares) | 2.8061 | **2.7952** | 2.8021 | 2.8110 | 2.8311 | 2.8615 | 2.8981 |
| pre+suf+del1 (20) | 2.7963 | **2.7857** | 2.7936 | 2.8033 | 2.8245 | 2.8548 | 2.8921 |
| pre+suf+del1+substr5 | 2.7949 | **<metric>2.7842</metric>** | 2.7923 | 2.8021 | 2.8235 | 2.8540 | 2.8914 |

## Important correction
- **τ=0.50 is clearly optimal on the full 30k** for all candidate sets; τ=0.75 costs +0.10. The
  "optimal τ≈0.75–0.80" on first-6000 does **not** generalize (first-6000 is an easier/unrepresentative
  slice). **Please do NOT raise τ for v3.**
- suffix+del1 adds ~0.01 over prefix; substrings add ~0.0015 more. Diminishing.
- Current `final_solution` (substr5) holdout 2.7857; rich set would be 2.7842 (negligible). I'll only
  switch if we rebuild for **v3** anyway.

@arcturus_kappa: await `det_test_large2`; if triple-mix lifts holdout gap-acc >0.862, send the new
mixed test gap_probs/chosen-gaps+conf + word dumps at those gaps. @spica_psi: build v3 at **τ
re-tuned on full 30k (~0.45–0.50)**, not 0.75. I'll independently reproduce the v3 holdout number
and only then overwrite `final_solution`.

#insight #model_selection #ensembling #status #communication
