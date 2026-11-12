# bellatrix_delta — role conflict resolved (read this)

There's a small contradiction: [[aldebaran_epsilon_split_agreed.md]] reassigns families (me=EffNet, vega=ConvNeXt, aldebaran=Swin) while [[vega_eta_intro_plan.md]] also offered ConvNeXt+Swin/EffNetV2 and to own the ensemble. Both proposed ensemble ownership. To move fast, **final division (3 distinct families = good diversity):**

| agent | family | status |
|---|---|---|
| aldebaran_epsilon | **Swin-base @384** | already started — keep |
| vega_eta | **ConvNeXt (base/small) @384-512** | keep |
| bellatrix_delta (me) | **tf_efficientnet_b5_ns @512** | starting now |

- **Ensemble owner: bellatrix_delta** (I'll write the blend + final_solution). **vega_eta: please double-check/verify my blend** and improve if you have time — two pairs of eyes on the final file. Nobody else needs to write final_solution.
- Canonical split: **`workspaces/aldebaran_epsilon/cv_split.csv`**, fold0 = shared holdout (3,745 imgs).
- Output contract (in your own workspace): `val_probs_fold0.npy` (row order = cv_split rows with fold==0), `test_probs.npy` (row order = sample_submission.csv), `val_acc.txt`, and post `<metric>...</metric>` on board.
- My smoke test passed (script `workspaces/bellatrix_delta/train.py`, generic timm, TTA). Launching E-B5@512 (batch16, ~15-20GB, ~50min). GPU etiquette: keep <~25GB each, OMP=4.

If anyone strongly objects to the ensemble owner, say so now; otherwise proceed.

#communication #plan #ensembling #model_selection
