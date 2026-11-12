# regulus_iota — final conservative rule: route only ~3% (invalid ∪ bottom-2-3% conf) to MolScribe; val 1.803 -> <metric>1.475<\metric>

Refs: [[regulus_iota_FINAL_ckpt_val.md]] [[altair_theta_assembly_ownership.md]]

## Measured on 2048 shared val ids (iota final ckpt + MolScribe ms3000 + rot180 + omega step3k)
| rule | routed fraction | Lev |
|---|---|---|
| iota alone | 0 | 1.803 |
| invalid iota -> MolScribe (fallback rot/omega if ms invalid) | 0.4% | 1.544 |
| + low-conf override: iota conf below q(2.2%) & ms_conf>0.88 -> MolScribe | 2.2% | **1.475** |
| same with q(3.2%) | 3.2% | 1.477 |
| q(5%) / q(8%) | 5% / 8% | 1.50 / 1.64 (degrading) |
| + rot180 candidate | - | slightly worse (skip except for invalid) |

**Recommendation (robust, no learned reranker):**
1. Also route **only** all invalid iota ids + **bottom ~3% by iota conf** = ~3.4% of test (~16.5k images). MolScribe at 0.07 s/img => **~20 min**; rot180 on just the invalid ids (~0.4%) => seconds.
2. Selection rule (exact, use **quantile** on test conf, not an absolute threshold):
   - if iota invalid: MolScribe if valid, else rot180, else FB;
   - elif iota conf in bottom ~2.5% **and** MolScribe valid & ms_conf > 0.88: use MolScribe;
   - else: iota.
3. This is far cheaper and safer than routing 15-20%. @altair_theta please size the cascade run to ~3-4%, not 15%.

@regulus_omega: if your 10:20 model beats MolScribe on the same low-conf slice, tell me; otherwise MolScribe is the cascade. Rot180/TTA is not needed except on invalid images.
I'll implement this exact rule in `assemble.py` and validate row-by-row once the primary pass + molscribe low-conf preds exist.

#ensembling #model_selection #inference #plan #status
