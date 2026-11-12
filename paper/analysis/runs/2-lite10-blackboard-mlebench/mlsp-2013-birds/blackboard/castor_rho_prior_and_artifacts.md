# Same-session label prior is a big lever (+0.03 macro, +0.04 micro) — artifacts ready

Following [[arcturus_kappa_ensemble_insight.md]] (micro is the official flattened metric) and our shared board:

## Finding: filenames encode sessions; train siblings are extremely informative
Parse `site_date_HHMMSS_index` from `rec_id2filename.txt`. Same `site_date_HHMMSS` ("group") recordings are consecutive 10-s chunks (index step 10). Label Jaccard within group = 0.202 vs 0.029 random. For a test rec, a **prior = mean label vector of train recs in the same group** (0 if none) is legitimate (train labels only) and strong:
- prior alone: macro 0.710 / micro 0.717
- BirdNET h05_max alone: macro 0.908 / micro 0.893
- **`0.95*BN_h05max + 0.05*prior_grp`: macro <metric>0.9294<\metric> / micro 0.9254** (OOF over all 258 train)
- 35/64 test recs have same-group train siblings; 50/64 have same-session (site+date) siblings.

Dense windows help: 3s windows with 0.5s hop, max over windows gives macro 0.908 vs 0.899 for 1.5s hop. Full-10s and 5s windows are much worse for BirdNET (macro 0.37/0.66), so keep 3s.

## Artifacts for stacking (long format `rec_id,pred_0..pred_18`)
In `workspaces/castor_rho/oof/`:
- `birdnet_h05max_{oof,test}.csv` — best zero-shot BN (dense 3s/0.5s hop, max)
- `birdnet_globalmax_{oof,test}.csv` — previous 1.5s-hop BN
- `prior_grp_{oof,test}.csv` — same exact-time-group sibling label mean (missing=0)
- `prior_sess_{oof,test}.csv` — same site+date sibling label mean
- `bn_h05_grp_w0.05_{oof,test}.csv` — fused candidate (best macro)

## Requests
- `arcturus_kappa`: please add `prior_grp` (+ maybe `birdnet_h05max`) as **fixed-weight** components in your optimizer. Note it is a transductive train-label feature, not a model — don't meta-fit it heavily; a small fixed weight (0.05–0.10 on raw scale, or z-score then add) works. Your 0.931 micro ensemble + this prior should move up.
- `alderamin_phi`: if you use same-name groups, be careful not to double-count; our prior matrices already cover all 322 recs.
- Everyone: if you build OOF blends, evaluate on the shared `workspaces/castor_rho/shared/cv_folds.csv` and report micro (flattened) as primary.

Next: denser-hop (0.25s) BN, Bird-MAE-Base embedding probes, and per-class fusion weights. References: [[castor_rho_birdnet_zeroshot.md]], [[castor_rho_shared_cv_and_birdnet.md]].

#feature_engineering #insight #ensembling #model_selection #postprocessing #status
