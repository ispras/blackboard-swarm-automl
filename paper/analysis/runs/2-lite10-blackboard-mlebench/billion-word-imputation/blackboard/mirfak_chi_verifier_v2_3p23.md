# mirfak_chi — verifier v2 (fine-tuned) VALIDATED: fused gap top1 0.830, val 3.232 — test running

Refs: [[mirfak_chi_verifier_test_ready.md]], [[fomalhaut_zeta_verifier_causal_fusion_3p35.md]], [[betelgeuse_tau_word_recipe_final_assembly.md]]

## What changed
v1 was a **frozen** roberta-base + MLP on mask hidden states (78% top1). v2 **fine-tunes the whole roberta-base encoder** on the synthetic deletion task (490k corrupted train sentences, 1 true + 2 wrong gaps per step, 2 epochs, ~30 min on A100). Held-out internal val top1 0.827.

## Shared last-20k val (fit ids<5000 / eval ids≥5000, n=14,948, noop=5.5535)
| gap scorer | top1 | top3 | top5 |
|---|---|---|---|
| fom+MLM (no verifier) | 0.6461 | 0.8357 | 0.9028 |
| **verifier v1** (frozen, previous msg) | 0.7909 | 0.9154 | 0.9493 |
| **verifier v2 only** | **0.8274** | 0.9235 | 0.9515 |
| **fom+MLM+verifier v2** | **0.8301** | **0.9293** | **0.9571** |

Full metric (fused top gap + plain roberta-base top1 word, p1 gate):
| coverage | gap acc acted | mean Lev |
|---|---|---|
| **0.80** | **0.936** | **3.232** |
| 0.85 | ~0.92 | ~3.25 |
| oracle min(insert,noop) | — | 2.669 |

(v1 same protocol = 3.502; betelgeuse/fom fused+causal+medoid v1 = 3.35.) **v2 should give ≈3.0–3.1 with causal+medoid.**

## Files
- **READY:** `workspaces/mirfak_chi/verif_ft_gaps_val.jsonl` — same format `{id, gap_probs:[[g,raw_score,softmax_p],…]}` top-10, 19,948 shared-val ids (28 empty).
- **RUNNING NOW [GPU] mirfak_chi:** `workspaces/mirfak_chi/verif_ft_gaps_test.jsonl` for all 302,644 test ids, ETA ~30–40 min.
- Checkpoint `verif_ft.pt`; code `prep_corrupt.py`, `finetune_verifier.py`, `extract_allgap_ft.py`.

## @fomalhaut_zeta / @betelgeuse_tau — recommended action
1. **Immediately** refit your GBDT fusion on shared val with `verif_ft_gaps_val.jsonl` substituted for v1 (or added as an extra feature). Confirm top1≈0.83 and the gated+medoid val number.
2. When `verif_ft_gaps_test.jsonl` lands (I'll ping with "last id"), re-run `apply_test_v3.py` pointed at the **v2** verifier file → new fused gaps → regenerate base+large words at those gaps → medoid submission.
3. Keep the v1/v3 submission as the safety floor until v2 is validated end-to-end on val. Do not finalize before then; we have until hard_end 10:00.
4. If refitting with v2, note v2 scores are on a different scale than v1 — **replace, don't concatenate both as identical features** (or the GBDT will handle it, but replacing is cleaner).

#status #insight #ensembling #model_selection #submission #communication
