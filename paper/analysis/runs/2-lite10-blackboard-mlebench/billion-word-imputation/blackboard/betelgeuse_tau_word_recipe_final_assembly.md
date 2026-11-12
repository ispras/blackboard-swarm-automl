# betelgeuse_tau — word-model upgrade recipe (use this in the final 3.3x submission)

Refs: [[fomalhaut_zeta_verifier_causal_fusion_3p35.md]], [[mirfak_chi_verifier_validated_3p50.md]], [[betelgeuse_tau_test_submission_4p19.md]]

Holding my 4.15 submission as you asked — **do not stop at 3.35 if my word recipe is better.** I measured the word side on the shared val at the **true gap** (20k) with our mask-MLM word distributions; the final word string matters a lot once the gap is right:

| word policy at true gap | mean Lev |
|---|---|
| roberta-base top1 | 2.261 |
| roberta-base medoid (my prod: base+fom, mpb .2) | 2.190 |
| **roberta-large medoid (alone)** | 2.054 |
| roberta-base + roberta-large (0.3/0.7) medoid | 2.036 |
| **0.3·base + 0.7·large + 0.4·fomalhaut-trigram + medoid, mpb=0.05** | **1.999** |

End-to-end (same chosen gaps, shared-val all 19,948): old word policy **4.202 → 4.153**; so this recipe is worth ≈ **−0.05** on top of the gap fusion. Both models are cached locally (`roberta-base`, `roberta-large`, HF, MIT).

## Exact medoid recipe (drop-in for your assembly)
At chosen gap `g`, from top-k conditional logp at that gap:
```
P = 0.3*exp(base.lp) + 0.7*exp(large.lp) + 0.4*P_fom_trigram(g)   # raw weights, not normalized
pool = {full words} ∪ {prefix len 1,2,3 of each word} ∪ {",","the","a","of","to","and","in","."}
x* = argmin_x  sum_w P[w]*Lev(x,w) + 0.05*(len(x)+4.57)
insert x* at g (or no-op if gated out)
```
My exact val artifact/code: `workspaces/betelgeuse_tau/val_lm_true.jsonl`, `tune_lm_word2.py`.
I also have **roberta-base+large at the current test chosen gaps** (`test_lm_chosen.jsonl`) but those gaps will change with the verifier fusion, so **regenerate with your final chosen gaps** (my `workspaces/betelgeuse_tau/lm_gaps.py --gaps <id,gap>.tsv --src test --out ...` runs both models at given gaps; ~10–13 min for ~230k gaps on A100 — @fomalhaut_zeta you offered the large-model run, use this script if useful).

## Coordination / final assembly (avoid two colliding writes)
- `@fomalhaut_zeta`: please deliver **`fused_gap_test.jsonl`** = per-id `{id, gap_probs:[[g,p],…] top-10}` from your full LightGBM (verifier+causal+ngram+MLM). I will consume it, regenerate base+large words at the chosen gaps, apply the medoid recipe, and write `final_solution/submission.csv`. If you prefer to assemble yourself, just use the recipe above and announce before overwriting — whichever we validate higher on the **identical val protocol (fit ids<5000 / eval ids≥5000)** wins. My current 4.15 CSV stays as the safety floor until then (do not delete it).
- `@mirfak_chi`: `verif_gaps_val.jsonl` already gives fomalhaut 3.35; get `verif_gaps_test.jsonl` finished — that is the critical path. Ping when the last id is written.
- No-op fallback and everything else unchanged.

#status #ensembling #communication #submission #model_selection
