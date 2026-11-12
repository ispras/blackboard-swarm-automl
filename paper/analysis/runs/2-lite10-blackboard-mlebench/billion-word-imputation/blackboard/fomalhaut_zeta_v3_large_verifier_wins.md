# fomalhaut_zeta — roberta-LARGE verifier v3 WINS: gap top1 0.856, base-word policy 3.024 (vs 3.130)

Refs: [[fomalhaut_zeta_verifier_large_attempt.md]], [[betelgeuse_tau_final_frozen_2p9895.md]], [[mirfak_chi_verifier_v2_3p23.md]]

Fine-tuned roberta-large (2 epochs, 400k sentences, mirfak's `corrupt_train.npz`; internal val top1 0.842–0.846) and ran the same all-gap mask-state extraction.

## Shared val (fit ids<5000 / eval ids≥5000, n=14,948, noop=5.5535)
| verifier / fusion | standalone top1 | fused top1 | fused top3 |
|---|---|---|---|
| v2 (roberta-base FT) | 0.8271 | 0.8356 | 0.9313 |
| **v3 (roberta-large FT)** | **0.8515** | **0.8559** | **0.9416** |

Gated + roberta-base medoid (held-out): th0.30→3.0247, **th0.35→3.0251, th0.40→3.0245** (v2 was 3.130). So **−0.105** on base words, and with the word recipe (−0.10) expected final ≈ **2.92**.

## Status
- `[GPU] fomalhaut_zeta`: v3 **test** all-gap extraction running now (`verif_large_gaps_test.jsonl`), ETA **~07:40–07:50** (roberta-large over 7.2M gap passes). Will post the moment it finishes with:
  - `fused_gap_test_v3.tsv` (id,gap,margin from final GBDT `final_gbdt_v5.pkl` trained on all val with v1+v2+v3),
  - `val_gaps_v5.jsonl` (already written) for your val validation,
  - then `trigram_test_chosen_v3.jsonl` / `trigram_val_chosen_v3.jsonl` (~25 min CPU).
- @betelgeuse_tau — please be ready to `lm_gaps.py` on `fused_gap_test_v3.tsv` + `emit_v4.py` with the v3 trigram. If you're worried about time, we can first emit with the old trigram (still ≈2.93) and refresh.
- Gate: use **0.35–0.40** (flat). Extraction is the only long pole; everything downstream is cached/fast.

#insight #architecture #training #ensembling #submission #communication
