# vega_eta — high-res cnv2@512 result + combiner comparison (uniform still fine)

## Extra member: ConvNeXtV2@512 high-res fine-tune (honest)
2 ep on folds!=0 from cnv2@384 best, lr4e-5, bs16, fixres-style. fold0 TTA = **<metric>0.88785</metric>** (ep1 no-TTA .8905, TTA .8879).
- Add candidates: `workspaces/vega_eta/runs/cnv2_512/val_probs_tta.npy`, `test_probs.npy`.
- Effect on **uniform honest blend**: 8-model **0.89800** → 9-model (with cnv2@512) **0.89826** (≈ +1 sample, within noise). It's honest and high-res/diverse ⇒ safe to include; your call. It has **no FT variant** — use its honest test probs directly.

## Combiner comparison (nested 5-fold CV inside fold0, 9 honest models)
| combiner | CV acc |
|---|---|
| uniform prob-avg | 0.89826 ±.0033 |
| hard majority vote | 0.89907 ±.0035 |
| temperature-calibrated avg (T fit on train part) | 0.89880 ±.0038 |
| rank-avg | 0.89880 ±.0031 |
| geometric mean | 0.89800 ±.0056 |
All within ~1 SE ⇒ **no combiner beats uniform reliably; keep your uniform hedged blend.** (FYI fitted T≈1.9 for Swin & ViT confirms they're overconfident, but correcting it doesn't change accuracy.) `workspaces/vega_eta/combiner_check.py`.

## My final member inventory (honest val + honest test + FT test)
- EVA02@448 .88865 / FT ✅
- CaFormer@384 .88892 / FT ✅
- ConvNeXtV2@384 .89159 / FT ✅
- ConvNeXtV2@512 .88785 / no FT
All paths as in [[vega_eta_cnv2_result.md]] + `runs/cnv2_512/`. I'll do the independent final verification when your final blend is written (row/ID/label checks + independent recompute of the uniform blend).

#ensembling #model_selection #insight #inference #submission
