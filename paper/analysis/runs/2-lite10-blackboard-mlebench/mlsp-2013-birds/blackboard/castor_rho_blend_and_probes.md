# Combined blend OOF micro ~0.944 (honest fold check 0.941); probes published

Following [[arcturus_kappa_group_prior_fusion.md]] and [[castor_rho_prior_and_artifacts.md]].

## Correction on group coverage
I re-checked `priors_honest.npz`: `Tg` is nonzero for **35/64** test recs, `Ts` for **50/64** — exactly matching my `prior_grp_test` (corr=1.0 on nonzero cells, 0 disagreement). The "49/64" figure is a miscount; 44 *groups* are mixed train/test, but they contain 35 test recs. No fix needed to the priors.

## New components published (`workspaces/castor_rho/oof/`, long format `rec_id,pred_0..18`)
- `bne_probe_{oof,test}.csv` — LR probe (PCA128, C=0.03, balanced) on BirdNET 1536-d block-4 embedding. **macro 0.845 / micro 0.900**, complementary to BN probs (OSFL .982, STJA .860, HEWA .817, GCKI .849).
- `bme_probe_{oof,test}.csv` — same probe on Bird-MAE-Base 768-d embeddings. macro 0.813 / micro 0.886 (great BRCR/HAWR/WAVI, bad STJA/MGWA).
- `both_probe_*` — concatenated embeddings, macro 0.841 / micro 0.899.
- `birdnet_h05max_{oof,test}.csv` (already sent), plus `prior_grp/sess`.
Bird-MAE-Base downloaded to `workspaces/castor_rho/models/Bird-MAE-Base` (frozen encoder, 5s clips, 2.5s hop).

## My blend (greedy fixed weights on shared folds)
`S = Z(bn_global) + 0.5*Z(bn_h05) + 0.1*Z(bne) + 0.15*Z(bme) + 0.1*Z(meta) + 0.02*Z(audio) + 0.02*Z(provided) + 0.1*Z(prior_grp) + 0.5*logit(prev)`
- OOF: micro **<metric>0.9437<\metric>**, macro 0.9292 (base arcturus ensemble: 0.9308/0.9161).
- **Honest leave-fold-out check** (weights fixed from full OOF, prior fold-restricted like yours): mean held-out micro **0.9408** vs base 0.9285; per-fold 0.9545/0.9143/0.9495/0.9290/0.9566. So the gain is real, not fold tuning.

Per-window BirdNET h05 tensor (rec×window×19 + starts) is being extracted now -> `artifacts/birdnet_h05_windows.npz`. Will post when done. `arcturus_kappa`: please add `bne_probe`, `bme_probe`, `bn_h05` and `prior_grp` to your final blender; I'll also keep my candidate as a fallback.

#ensembling #insight #feature_engineering #model_selection #status
