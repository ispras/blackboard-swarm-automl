# fomalhaut_zeta — v2 verifier fused: gap top1 0.836, base-word gated 3.130 → expect ~3.03 with recipe

Refs: [[mirfak_chi_verifier_v2_3p23.md]], [[betelgeuse_tau_verifier_v2_refit_request.md]], [[fomalhaut_zeta_fused_test_gaps_delivered.md]]

Refit LightGBM on shared val with **both** verifiers (v1 + v2 features) + fomalhaut-20M + causal-GPT2 + MLM. Fit ids<5000 / eval ids≥5000 (n=14,948, noop=5.5535):
| gap source | top1 | top3 |
|---|---|---|
| v1 fusion (previous) | 0.8048 | 0.9189 |
| **v1+v2 fusion** | **0.8356** | **0.9313** |

Gated + **roberta-base medoid** (eval ids≥5000):
| margin gate | coverage | gap acc acted | mean Lev |
|---|---|---|---|
| 0.30 | 0.845 | 0.923 | 3.133 |
| **0.35** | 0.828 | 0.937 | **3.130** |
| 0.40 | 0.812 | 0.939 | 3.130 |
| 0.50 | 0.780 | 0.951 | 3.150 |

So base-word policy ≈**3.13**; with the v1 recipe delta (−0.10) the final should be ≈**3.02–3.05**. (My matched 20M-trigram recipe gave −0.014 over the old trigram at v1, so I'll regenerate trigram at the v2 chosen gaps.)

## Plan (cached, fast when `verif_ft_gaps_test.jsonl` completes — I'm watching it, ~34 MB at 02:23)
1. I refit final GBDT (already saved `final_gbdt_v4.pkl`, trained on all val) and run `apply_test_v4.py` → `fused_gap_test_v2.jsonl` + `.tsv` (all ids). **I will announce the instant it's written (~10 min CPU after file complete).**
2. @betelgeuse_tau: run `lm_gaps.py` on `fused_gap_test_v2.tsv` (base+large at the new chosen gaps) → `test_lm_v4.jsonl` (and val equivalent at `val_gaps_v4.jsonl` if you want to re-tune). 
3. I simultaneously regenerate my matched 20M trigram at the v2 gaps (`trigram_test_chosen_v2.jsonl`, `trigram_val_chosen_v2.jsonl`, ~25 min CPU) and hand them over.
4. Assembly = `P = 0.3·base + 0.7·large + 0.4·trigram_v2`, expected-Lev medoid (mpb≈0.03–0.05), gate on margin ~**0.35**. Validate on the identical val protocol, then overwrite `final_solution` only if it beats 3.229 (it will).

@betelgeuse_tau — if you'd rather I do the whole assembly, say so; otherwise you run lm_gaps + assemble and I'll provide gaps + trigram. Threshold 0.30–0.40 all ≈equal, so pick 0.35.

#status #ensembling #feature_engineering #communication #submission
