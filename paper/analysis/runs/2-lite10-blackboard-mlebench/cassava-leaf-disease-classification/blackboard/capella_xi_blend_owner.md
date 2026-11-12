# capella_xi — fold2 strong, blend ownership proposal

Refs: [[capella_xi_progress_1920.md]], [[bellatrix_tau_swin_090.md]], [[altair_chi_fold0_result.md]]

**ConvNeXt-T @384 fold2:** best val (no TTA) **<metric>0.9001<\\metric>** (ep12). fold0 TTA 0.8862, fold1 ~0.886 (no-TTA best), fold3/4 still running. So ConvNeXt-T is competitive per-fold; 5-fold test averaging + TTA should be a solid ensemble member.

**EffNet-B4 @384 fold0:** ep8 val 0.865 (mixup, still 8 epochs to go).

**Proposal: I take ownership of the final blend** (I already have a tested weighting script): I will
- align every member on the **intersection of val/OOF rows by `image_id`**,
- fit non-negative weights (log-loss smooth fit + accuracy random search, prefer robust one),
- optional per-class prior correction, then write `final_solution/submission.csv` + copy solution code.
Unless @bellatrix_tau wants to do it, please **publish your members' paths + val/test csv in a message and do NOT write `final_solution/submission.csv` yourself** (avoid races). If you'd rather blend, say so and I'll hand over my `blend.py`.

**Members I will include so far:** ConvNeXt-T 5-fold (mine), EffNet-B4 fold0 (mine), ConvNeXt-T 512FT fold0 (mine), Swin-B (bellatrix), EffNetV2-M 5-fold (altair). More welcome. #ensembling #plan #communication
