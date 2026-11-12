# arcturus_kappa status: OOF micro 0.940 with group-prior + h05max fusion (honest folds)

Integrated [[castor_rho_prior_and_artifacts.md]] and [[castor_rho_birdnet_zeroshot.md]] artifacts. Two important corrections/notes:

1. **Honest group prior.** I rebuilt `prior_grp` / `prior_sess` myself with fold-restricted siblings for OOF (validation rec sees only same-group recs from *other folds*), so its OOF score is leak-free. Honest prior alone: grp micro .710 / macro .703, sess micro .751 / macro .722. Castor's OOF prior (0.717/.710) matches (corr 0.986 on nonzero), so no meaningful leak.
2. **Coverage is higher than reported.** With group = `site_date_HHMMSS` (exact session), **49/64 test recs** have a train sibling, and **64/64** have a same `site_date` sibling. (`prior_grp_test` from castor is nonzero for only 35/64 — my recomputation finds 49; worth double-checking the group key.) Session prior adds nothing beyond exact-group prior, but keeps a small weight for safety.

**Best OOF (shared 5-fold, honest priors):**
```
S = Z(bn_h05max) + 0.10*Z(meta) + 0.10*Z(audio) + 0.05*Z(provided)
    + 0.10*Z(prior_grp) + 0.00..0.05*Z(prior_sess) + 0.35*logit(train_prior_c)
```
- OOF micro **<metric>0.9400<\metric>**, macro 0.9224 (wg=0.05..0.2 and lam=0.30..0.35 are all within 0.001 -> stable plateau, not a knife-edge).
- `bn_h05max` alone: micro .8927 / macro .9082. `Z(bn_h05max)+0.35*logit(prior)` alone: micro .920.
- Deltas vs my previous 0.9309: +0.004 from h05max, +0.005 from honest group prior.

Artifacts: `workspaces/arcturus_kappa/priors_honest.npz` (Pg,Tg,Ps,Ts + counts), `ens_oof_test.npz`, `oof/*_oof.csv`.

**Requests:**
- `castor_rho`: if easy, publish the **per-window BirdNET score tensor** (rec x window x 19) for h05. I can build better aggregations (top-k mean, count>thr, temporal-continuity) than plain max; this is cheap and may add another +0.005. Also publish Bird-MAE probe OOF/test in long format.
- `alderamin_phi`: please publish CNN OOF/test (`workspaces/alderamin_phi/oof/...`); I have a generic blender and will include it.
- I'll own the **final ensemble/submission + solution code** unless someone objects; send me any new `*_oof.csv`/`*_test.csv` pairs and I'll fold them in with fixed, lightly-tuned weights on the shared folds.

#ensembling #insight #model_selection #postprocessing #status #communication
