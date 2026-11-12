# capella_kappa — round-2 slot claim + status

Refs: [[achernar_delta_convnext_fold0.md]], [[capella_kappa_tools_endgame.md]]

Great, ConvNeXt-small fold-0 <metric>0.8932<\metric>. Agreed plan:
- achernar: `convnext_base` 5-fold (or FixRes@512) — your call.
- arcturus: EffNetV2 + finish effb4.
- **me (capella): Swin family + a ViT for transformer diversity.**

## My concrete plan
1. Finish Swin-Base@384 fold-0 (~35 min left, ep4 val <metric>0.8868<\metric> — plateauing around 0.887-0.890).
2. Re-run inference with multi-scale + h/v-flip TTA (`infer.py`).
3. Round-2a: **`vit_base_patch16_384.augreg_in21k_ft_in1k` @384** single-fold (fold 0) — different transformer inductive bias than Swin.
4. Round-2b (if time): full-data (100%) Swin for stronger test preds.

Artifacts: `workspaces/capella_kappa/swin_base_384/` (fold-0 `val_probs.npy`, `test_probs.npy`), then `infer_swin/`, `vit_base_384/`.

I'll announce before launching each (we're at 3 heavy jobs once achernar starts fold1 and I start ViT; keep ≤3).

#plan #communication #architecture #model_selection #status
