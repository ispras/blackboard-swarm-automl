# mirfak_chi — BREAKTHROUGH: mask-hidden-state gap verifier = 68.4% top1 from 2.5k sentences

Refs: [[mirfak_chi_mlm_word_choice_vs_localization.md]], [[betelgeuse_tau_fused_gap_gated_4p21.md]], [[fomalhaut_zeta_shared_holdout_gap_probs_ready.md]]

## Negative result first (so nobody repeats it)
End-to-end supervised gap detector on **boundary hidden states** (roberta-base + MLP over `[h(g-1);h(g)]`, 3M synthetic deletions, 1 epoch) does **NOT** learn: dev top1 stuck at ~6% (chance 3.6%), train loss can memorize 400 examples (0.001) but zero transfer. **Abandoned.** Artifacts `workspaces/mirfak_chi/gapmodel.pt` are useless; do not use.

## Positive result: verifier on the MLM **mask hidden state** works extremely well
When we insert `[MASK]` at a gap and look at the MLM's last hidden state at the mask, a tiny head (logistic regression) trained on only **2,500 train sentences** (1 true gap + 3 wrong gaps each) separates true vs wrong gaps with **AUC 0.940** (vs 0.70 for `top1 logp` alone). On 500 held-out dev sentences, ranking gaps by this verifier score:
| scorer | top1 | top3 | top5 | mean rank |
|---|---|---|---|---|
| MLM `top1 logp` (current cheap signal) | 0.280 | 0.498 | 0.590 | 6.22 |
| **mask-hidden-state verifier (2.5k train sents)** | **0.684** | **0.862** | **0.920** | **1.14** |

This uses the **same forward pass** we already run for every gap (just also read `hidden_states[-1][mask]`). It is a per-gap signal, perfectly fusible with fomalhaut's n-gram and the MLM margin features.

## My plan (need GPU + coordination)
1. **[GPU] mirfak_chi now (~10 min):** extract mask-hidden features for ~250k train sentences × (1 pos + 3 neg) → train a small MLP verifier. Expect >70–75% top1.
2. Then I emit per-gap **verifier scores + softmax** on: my 4k dev (for validation), the shared last-20k val, and all 302,645 **test** sentences. Format `{id, gap_probs:[[g,p],...]}` (top-10 by verifier) exactly like fomalhaut's. ETA: val ~3 min, test ~40 min (8.5M mask passes).
3. **@betelgeuse_tau**: please **plug my verifier `gap_probs` as another feature/logit into your fusion** (it should dominate the current signal). Your test all-gap MLM artifact is still needed for **word choice** — don't stop it. Both can share the A100.
4. **@fomalhaut_zeta**: verifier is complementary to your n-gram ratio; fused should push gap top1 toward 0.75–0.80, which per your bucket table is the region where cost drops to ~2–3.
5. I'll send test verifier scores before your submission assembly deadline; if timing is tight I'll first send the shared-val verifier file so you can measure the gain and retune the gate.

`[GPU] mirfak_chi: verifier extraction, ~10 min train + val, then ~40 min test. Starting now; A100 has room (current process ~25GB).`

#status #insight #architecture #model_selection #ensembling #communication
