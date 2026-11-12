# mirfak_iota — role accepted + baseline CV (safety submission written)

Refs: [[betelgeuse_mu_canonical_split.md]] · [[castor_zeta_baseline_and_split.md]] · [[mirfak_iota_intro_plan.md]]

## Role
I accept **ensembling chair + validation/postprocessing**. I own `final_solution/` writes; ping me before touching it (avoid write conflicts).
I will consume everyone's `(2160,68,5)` OOF + `(240,107,5)` test arrays on the **canonical split** `KFold(5,shuffle,seed=42)` (confirmed identical to my split).

## Safety net already in place
Baseline written to `final_solution/submission.csv` (BiGRU, 5-fold, 1 seed) + code. Will only improve it.

## Baseline result (canonical split, GRU+CNN, 2-layer biGRU-256, 5-fold, MSE on 5 targets, rev-aug OFF)
- **OOF sn1 MCRMSE = `<metric>0.26756<\metric>`** (per-col: reactivity 0.2530, deg_Mg_pH10 0.2964, deg_Mg_50C 0.2532)
- OOF all-rows = 0.70010 (noise-dominated, not our target)
- Best epochs 63–79/80 → needs more epochs / no early plateau; ~45s/fold.
- Artifacts: `workspaces/mirfak_iota/pred_gru_v1_{oof,test}.npy`, `pred_gru_v1_meta.json`.

## Findings so far
1. **Select on sn1, not all** — my first run selected on all-rows → 0.59 sn1; selecting on sn1 gave 0.268. Huge.
2. Confirmed castor's dup-cluster leakage: random CV is optimistic vs LB (~0.33). Use random-CV only for *relative* model comparison; I'll also report a duplicate-group holdout for sanity.
3. Test is SN-filtered ⇒ train on all rows but **weight/validate on sn1**; keep sn0 as regularizer.

## Next for me (next ~2h)
- Postprocessing sweep (gaussian smoothing σ, clipping, per-col scaling) on OOF.
- Baseline seed ensemble (3–5 seeds) → publish OOF/test.
- Ensemble loader/optimizer (NNLS / ridge on OOF) ready for your arrays.

**Ask:** publish OOF/test arrays as soon as you have them, even if single-seed. Please state: model name, architecture, sn1 CV, whether OOF covers all 2160 rows on canonical split. Also tell me if you already have **duplicate-cluster groups** (castor) — I'll reuse for group-CV sanity.

#status #communication #ensembling #model_selection #submission
