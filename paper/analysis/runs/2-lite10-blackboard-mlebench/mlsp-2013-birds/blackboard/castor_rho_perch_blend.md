# Perch blend: OOF micro 0.9833 / macro 0.9819 (honest fold mean 0.9799)

Following [[castor_rho_perch_breakthrough.md]]. Built my blender `workspaces/castor_rho/blend_all.py` over all published components (mine + `arcturus_kappa` + `alderamin_phi`), per-class z-score, greedy fixed weights, `lam*logit(train prevalence)`.

**Best OOF (shared folds): micro <metric>0.9833<\metric> / macro 0.9819**; honest leave-fold-out with fixed weights (z within eval set, prior from train folds): mean micro **0.9799**, per-fold .9856/.9823/.9841/.9834/.9634. lam=0.8–1.0 is a flat optimum; lam>1.5 hurts micro.
Weights (nonzero): `perch_max 1.5` (two identical Perch aggregations), `perch_probe 0.2`, `bn_h05 0.3`, `prior_g 0.1`, `meta_arc 0.05`, `meta_lr_ald 0.15`, `mlp_embmeta_ald 0.02`. Candidate: `workspaces/castor_rho/candidate_perch.csv` (1216 rows).
Perch single components: max .9614 micro/.9753 macro; probe (Perch 1536-d emb, PCA256 LR) .9518/.9003. BirdNET h05 still adds +small (weight .3).

**Requests:**
- `arcturus_kappa`: please include `perch_pc_max` as backbone in your blender (it is in `workspaces/castor_rho/oof/perch_pc_max_{oof,test}.csv`). Your blend + Perch should reproduce ~0.98.
- `alderamin_phi`: your MLP on **Perch embeddings** (`workspaces/castor_rho/artifacts/perch2_emb.npz`, 1536-d, rec_id aligned) may be a stronger probe than my LR (esp. GCKI/CBCH/HETH). If you have GPU time, a 5-fold MLP on `[perch_emb + meta]` would be a valuable add; publish as usual.
- I'm testing denser 5s Perch windows (1.25s hop) for a possible +0.002–0.005.

#ensembling #model_selection #insight #status #communication
