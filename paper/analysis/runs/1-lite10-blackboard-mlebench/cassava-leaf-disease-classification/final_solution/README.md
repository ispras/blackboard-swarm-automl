# Cassava Leaf Disease Classification — swarm solution

Metric: **accuracy** (5-class). Test set: 2,676 images (`data/test_images`), submission `image_id,label`.

## Final method
**Equal-weight arithmetic mean of 11 fine-tuned models** (probability space), then argmax:

```
p_final = mean_i p_i              # one weight=1 per member
label   = argmax_c p_final[c]
```

Combiner: `code/make_final.py` (all members, `--out final_solution/submission.csv`).

### Why equal weights
On the shared fold-0 holdout (~3,745 imgs) we compared arithmetic-equal, geometric-equal,
majority vote and Caruana greedy weight search with nested CV. Learned weights **overfit**
(greedy 0.8977 vs equal 0.900-0.902); arithmetic/geometric/majority differ only within noise.
Split-half set selection always picked the full member set. So we use plain equal weights
(arithmetic mean) over all diverse members.

### Validation
Fold-0 holdout accuracy of the final equal-weight union =
**0.9012** arithmetic (0.9007 geometric); each member's fold-0
single-model accuracy is 0.879–0.895). Upstream models used 100% of train data for the test
predictions where available, so true test accuracy is expected >= this estimate.

## Members (all trained only on `data/train_images` + ImageNet-pretrained weights)
Canonical CV split: `StratifiedKFold(5, shuffle=True, random_state=42)` over `data/train.csv`
(`workspaces/shared/folds5_seed42.csv`). All members use h/v-flip TTA.

| member | family | data | fold-0 val |
|---|---|---|---|
| achernar_convnext | convnext_small (384, +mixup) | 5-fold avg + 100% | 0.895 |
| arcturus_cvnxb | convnext_base (384, +mixup) | mean(fold-0, 100%) | 0.893 |
| arcturus_v2s_best | tf_efficientnetv2_s (384, +mixup) | mean(5-fold, 100%) | 0.892 |
| arcturus_v2m480 | tf_efficientnetv2_m (480, +mixup) | fold-0 | 0.887 |
| arcturus_effb4 | efficientnet_b4 (512) | 2-fold | 0.879 |
| capella_swin | swin_base (384) | 100% | 0.892 |
| capella_swin_mix | swin_base (384, +mixup) | 100% | 0.889 |
| capella_vit | vit_base_patch16_384 | 100% | 0.890 |
| capella_vit_mix | vit_base_patch16_384 (+mixup) | 100% | 0.894 |
| capella_beit_mix | beit_base (384, +mixup) | 100% | 0.888 |
| capella_xcit_mix | xcit (384, +mixup) | 100% | 0.888 |

## Code
- `code/train_convnext.py` — achernar ConvNeXt trainer (5-fold / `--full`), multi-scale TTA inference in `code/infer_ms.py`.
- `code/train_arcturus.py`, `code/train_capella.py` — other agents' trainers.
- `code/assemble_oof.py` — assemble per-fold predictions into OOF/test arrays.
- `code/make_final.py` — the final ensemble builder (writes `submission.csv`).
- `code/ensemble_capella.py` — independent ensemble implementation (cross-check).

Honest protocol: trained only on the training set; validation/early-stopping on a holdout carved
from it (`folds5_seed42.csv`). No test labels or external data used.

## Erratum (v4, correctness fix)
`arcturus_effb4` (`effb4_512/test_probs_2fold.npy`) contained **NaN** for test rows 1436 and 1897.
In the arithmetic union this made those rows NaN -> argmax returned class 0. Row 1436
(`2441927798.jpg`) is called class 4 by 9/10 members, so it was corrected **0 -> 4**
(1 label changed). NaN rows were filled with the per-image mean of the other 10 members
(sanitized member: `workspaces/capella_kappa/effb4_512/test_probs_nanfixed.npy`) and the union
re-averaged. Final `submission.csv` md5 `e32f8872453c714315ec0d346409a01d`. Fold-0 holdout of the
union is unchanged (0.9012); the fix only removes a NaN-induced error.
