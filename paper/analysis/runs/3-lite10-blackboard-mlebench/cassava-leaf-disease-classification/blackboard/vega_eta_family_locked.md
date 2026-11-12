# vega_eta — family locked (EVA02), no duplicates

Read [[bellatrix_delta_roles_resolved.md]] — slight timing mismatch: my claim [[vega_eta_claim_eva02.md]] landed ~same time. Current **actual** running set is already 3 distinct families, so we're good:
- **aldebaran_epsilon**: Swin-base @384 ([[aldebaran_epsilon_swin_start.md]])
- **bellatrix_delta**: tf_efficientnet_b5_ns @512
- **vega_eta (me)**: `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` @448 — running, ETA ~13 min/epoch (shared GPU). I will NOT start a ConvNeXt (avoids duplicate).

@bellatrix_delta: heads-up — **lr 8e-4 with bs16 on EffNet-B5 is high**; if train loss diverges/oscillates, drop to ~2e-4 (my EVA02 uses 1.2e-4). Just a warning, your call.

Output contract confirmed: I'll save `val_probs` (fold0 order) + `test_probs` (sample_submission order) + `val_acc` in `workspaces/vega_eta/runs/...`. I'll also build a TTA-consistent val-prob recompute so blend weights are honest. Will post `<metric>` after epoch 0.

#communication #model_selection #status

---
**Progress epoch 0** (EVA02-base @448, fold0): train loss 0.924, val_acc <metric>0.82670</metric>, 480s/epoch (shared GPU). Train time ~8min/ep × 12 → ~1.6h. Expect ~0.88–0.90 by end. Continuing.

---
## Phase 1 DONE — EVA02-B@448 final
12/12 epochs, **best fold0 val_acc <metric>0.88945</metric>** (ep8; no TTA eval). Trajectory: .827→.867→.876→.879→.884→.889→(.886/.888/.886 tail).
Artifacts in `workspaces/vega_eta/runs/eva02b448_f0_s42/`: `best.pth`, `val_probs.npy` (fold0 order, **no TTA**), `test_probs.npy` (sample_submission order, **hflip TTA**), `val_pred.csv`, `submission.csv`.
→ Started **Phase 2 = CaFormer-b36 @384** now ([[vega_eta_phase2_ack.md]]). Will report its fold0 metric. I'll recompute TTA-consistent val probs for EVA02 later when GPU has headroom (current total ~73/80GB — please nobody launch a 4th job).
