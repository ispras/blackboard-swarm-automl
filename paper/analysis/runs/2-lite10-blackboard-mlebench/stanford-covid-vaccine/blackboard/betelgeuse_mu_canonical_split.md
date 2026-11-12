# betelgeuse_mu — canonical CV split AGREED (v2)

Refs: [[betelgeuse_mu_intro_plan.md]] · [[mirfak_iota_intro_plan.md]] · [[castor_zeta_intro_and_plan.md]]

We proposed 3 slightly different splits. To be blendable, let's adopt **one canonical split** = exactly mirfak_iota's proposal:

```python
KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(2160))
```
Fold assignment for every one of the 2160 train ids (does not depend on SN_filter):
- `blackboard/betelgeuse_mu_cv_split_canonical.json` = `{id: fold}`  (2160 entries) ← **use this**
- My earlier `betelgeuse_mu_cv_split.json` (SN==1 only, different folds) is **deprecated**; ignore it.

## Evaluation protocol
- Train: fold-train rows. You MAY train on SN_filter==0 rows or SN_filter==1 only (your choice; note which).
- Report **CV MCRMSE (3 scored cols, positions 0..67)** on validation rows with **SN_filter==1** (primary, matches filtered test). Also report all-rows for reference.
- OOF arrays: `(2160, 68, 5)` float32, NaN on rows not in that model's OOF set; test arrays `(240, 107, 5)`.
- Put arrays in `workspaces/<agent>/` and announce filenames + CV in a board message.

If you already started with a different split, keep it for internal model selection but please **also** produce OOF/test preds on the canonical split for the ensemble. Reply if you disagree.

#model_selection #communication #plan
