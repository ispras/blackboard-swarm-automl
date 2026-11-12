# Billion Word Imputation — FINAL swarm solution

`submission.csv`: 302,644 unique ids (0..302,643), sorted, `id,"sentence"`.
Task: one interior word (never first; last token `.`) was deleted from each test sentence;
insert it back to minimise the mean character **Levenshtein distance** (lower is better).

## Headline
- Shared-holdout estimate (last 20k lines of `data/train_v2.txt`; all fusors fit on ids<5000,
  evaluated on the 14,948 unseen ids >=5000): **≈2.88 mean Levenshtein**
  (no-op baseline 5.55; v1 4.19 → verifier-v1 3.23 → verifier-v1+v2 2.99 → roberta-large v7 2.92 → roberta-large-v3 fused 2.88).
- Official test score is computed by the coordinator after hard_end. No test labels used.

## Final pipeline
1. **Gap localisation** — a LightGBM ranker over per-candidate-gap features (built by
   `fomalhaut_zeta`), combining four independent signals:
   - mediated-bigram likelihood ratio `log(count(a,*,b)/count(a,b))` over 20M train lines
     (n-gram localizer);
   - **verifier v1** — MLP on the roberta-base `[MASK]` hidden state at each gap, trained on
     synthetic deletions (`mirfak_chi`);
   - **verifier v2** — the same idea with a fully fine-tuned roberta-base encoder
     (`mirfak_chi`);
   - **verifier-large v3** — the same head on a fully fine-tuned **roberta-large** encoder
     (fine-tuned by `fomalhaut_zeta` on 400k synthetic deletions ×2 epochs; standalone
     shared-val top-1 0.8515, the best single verifier and the one used in the final);
   - **causal GPT-2-medium surprisal** `-log P(token_after_gap | prefix)` and percentiles
     (`fomalhaut_zeta`), plus roberta-base masked-LM margin features (`betelgeuse_tau`).
   Shared-val: gap **top-1 0.856 / top-3 0.942** (final v5/v3 fusion, `final_gbdt_v5.pkl`).
2. **Gate** — insert only if the fused margin `>= 0.35`, else submit unchanged
   (test coverage 85.4%; shared-val coverage 85.6%).
3. **Word = expected-Levenshtein medoid** at the chosen gap (`betelgeuse_tau`):
   `P = 0.3·P_roberta-base + 0.7·P_roberta-large + 0.3·P_trigram-middle(at chosen gap)`,
   then insert the string `x` (full word, 1/2/3-char prefix, or common token) minimising
   `Σ_w P(w)·Lev(x,w) + 0.03·(|x|+4.57)`. Cap: this makes gap errors cheap and extracts
   partial credit from prefix matches.

## Key files
- `emit_v4.py` — final assembler (gaps + base/large word distributions + 20M trigram at the
  chosen gap → gated medoid submission).
- `lm_gaps.py` — runs roberta-base + roberta-large at a given `(id, gap)` list.
- `run_mlm_val.py`, `run_mlm_test.py` — all-gap roberta-base MLM statistics (val/test).
- `make_val.py`, `eval_submission.py` — shared holdout construction + exact-Levenshtein scorer.
- `train_fusion.py`, `predict_test.py` — previous (v1, 4.19) pipeline, kept for reference.

### Upstream component code (`upstream/`)
- fomalhaut_zeta: `gap_probs_unified.py` (mediated-bigram localizer), `causal_feats.py`
  (GPT-2-medium surprisal), `build_final_v4.py` / `apply_test_v4.py` (LightGBM gap fusion),
  `trigram_at_gaps_v2.py` (20M middle-word counts at chosen gaps), `eval_recipe_v4.py`,
  `emit_final.py`, `fused_gap_test_v2.tsv`.
- mirfak_chi: `prep_corrupt.py`, `finetune_verifier.py`, `extract_allgap_ft.py`
  (synthetic-deletion data + fine-tuned roberta-base verifier v1/v2 and test extraction),
  `v7_pipeline.py`, `v7_apply_test.py`, `fused_gap_test_v7.tsv` (v7 gap fusion),
  `build_final_v5.py`, `apply_test_v5.py`, `fused_gap_test_v3.tsv` (final v5/v3 gap fusion).
- betelgeuse_tau: `verif_large.pt` (roberta-large verifier), `val_v7_gaps.py`.

## Upstream artifacts (see blackboard for provenance/links)
- `workspaces/fomalhaut_zeta/fused_gap_test_v3.tsv` (final v3/v5 fused chosen gap + margin)
- `workspaces/betelgeuse_tau/test_lm_v3b.jsonl`, `workspaces/fomalhaut_zeta/trigram_test_chosen_v3.jsonl` (words at v3 gaps)
- `workspaces/fomalhaut_zeta/trigram_{test,val}_chosen_v2.jsonl` (20M middle-word counts at chosen gaps)
- `workspaces/mirfak_chi/verif_gaps_{val,test}.jsonl`, `verif_ft_gaps_{val,test}.jsonl` (verifier v1/v2)
- `workspaces/betelgeuse_tau/{mlm_test_*.jsonl, test_lm_v4.jsonl, val_v4_lm.jsonl}` (word distributions)

## External models (source logging)
| model | source | revision | license | purpose |
|---|---|---|---|---|
| `roberta-base` | HuggingFace Hub `roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | MIT | MLM per-gap features + word distribution; encoder fine-tuned for verifier v2 |
| `roberta-large` | HuggingFace Hub `roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | MIT | word distribution at the chosen gap |
| `gpt2-medium` | HuggingFace Hub `gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | MIT | causal surprisal gap feature |
| roberta-large verifier v3 | fine-tuned locally by `fomalhaut_zeta` from `roberta-large` on 400k synthetic train deletions ×2 epochs | `722cf37b1afa9454edce342e7895e588b6ff1d59` (base) | MIT | final gap verifier (mask-hidden-state), val top-1 0.8515 |
| roberta-large verifier (v7) | fine-tuned locally by `betelgeuse_tau` from `roberta-large` on 250k synthetic deletions ×1 epoch | same base | MIT | intermediate v7 verifier, val top-1 0.842 (not in final) |

Verifier v1/v2 heads/encoders were trained **only on synthetic deletions from `data/train_v2.txt`**.
No competition-specific checkpoints, notebooks, write-ups, or test labels were used. The holdout
is the last 20k lines of `data/train_v2.txt`; all early stopping/model selection used only it.
