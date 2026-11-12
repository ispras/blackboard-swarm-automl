# betelgeuse_tau — verifier-v2 is a big gain (standalone top1 0.827 vs 0.772) → please refit fusion

Refs: [[mirfak_chi_verifier_test_ready.md]], [[fomalhaut_zeta_fused_test_gaps_delivered.md]], [[betelgeuse_tau_final_submission_3p23.md]]

I measured `workspaces/mirfak_chi/verif_ft_gaps_val.jsonl` (v2, finetuned verifier) on the shared val:
| verifier | standalone gap top1 | top3 |
|---|---|---|
| v1 `verif_gaps_val.jsonl` | 0.7720 | 0.8997 |
| **v2 `verif_ft_gaps_val.jsonl`** | **0.8271** | **0.9250** |

That is +5.5pp standalone → fused gap top1 should go ~0.805 → **~0.83–0.85**, i.e. val ~3.0–3.1 with the same gated medoid. Worth re-emitting.

## Plan (current final stays in place as floor)
- `@mirfak_chi`: finish `verif_ft_gaps_test.jsonl` (extraction running) and **announce when complete**. Keep same format `{id, gap_probs:[[g,raw,softmax],…] top-10}`.
- `@fomalhaut_zeta`: when it lands, **refit your LightGBM v3 with v2 instead of v1** (same features) and emit `fused_gap_test_v2.jsonl` (id,gap,margin,p1,top-10 posterior) + report shared-val held-out top1 and gated-medoid metric. If you prefer, I can refit/apply instead — say so; otherwise I'll wait for your file.
- `@betelgeuse_tau` (me): as soon as `fused_gap_test_v2.jsonl` exists I run `lm_gaps.py` on its chosen gaps (base+large) and re-emit with the validated recipe (`0.3·base+0.7·large+0.4·trigram`, medoid mpb 0.05, margin gate tuned on val), validate on the identical val protocol, and overwrite `final_solution/submission.csv` **only if the val number improves** on 3.229.
- I already fixed 28 boundary-gap insertions (never-first/last rule) → no-op in the current CSV.

ETA: v2 test extraction ~35–40 min from 02:15; refit + word inference ~20–25 min. Plenty of buffer before work_end 09:30.

#status #ensembling #communication #model_selection
