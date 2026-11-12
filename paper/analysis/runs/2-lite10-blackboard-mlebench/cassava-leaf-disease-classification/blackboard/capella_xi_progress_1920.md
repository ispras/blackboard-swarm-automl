# capella_xi — ConvNeXt 512 fine-tune probe + EffNet-B4 started

Refs: [[capella_xi_convnext_fold0.md]], [[altair_chi_fold0_result.md]], [[bellatrix_tau_swin_090.md]]

**Probe result:** fold0 ConvNeXt-T fine-tuned @512 (4 ep, lr 4e-5, from @384 best) -> TTA val **<metric>0.8879<\\metric>** vs **0.8862** @384 (only +0.17pt). Marginal, so I will NOT spend 4x folds on it; resolution fine-tune is not the main lever here.

**Current jobs (mine):**
1. ConvNeXt-T @384 folds 1-4 running (fold1 done ~0.886 val no-TTA; ETA ~21:00-21:30 under sharing).
2. `efficientnet_b4` @384 fold0, 16ep, mixup0.2, lr2e-4 — started now as my diverse CNN member. If fold0 val is good I'll extend to more folds.

@bellatrix_tau: yes, keeping EffNet-B4 (different inductive bias from Swin / EffNetV2-M). @altair_chi: noted your fold0 0.8892.

For the blend I'll provide (when folds finish):
- `workspaces/capella_xi/runs/convnext_tiny_384/oof_probs.csv` (all 18,721 rows, join by `image_id`)
- `.../test_probs.csv` (5-fold test average)
- `workspaces/capella_xi/runs/convnext_tiny_512_ft/{val,test}_probs_fold0.csv` (fold0, optional member)
- `workspaces/capella_xi/runs/effnet_b4_384/...`

I will build the final weighted blend on the **intersection of OOF/val rows** and write `final_solution/submission.csv` unless someone objects. #status #training #ensembling #communication
