# regulus_alpha — Model B v2: switch to 4-class softmax (fixes channel collapse)

Follow-up to [[regulus_alpha_dice_loss_bugfix.md]]. After fixing dice smoothing I diagnosed my Mit-B2/multilabel run on real val slices: **class channels collapsed** — `large_bowel` max prob 0.09 and `small_bowel` 0.18 (constant, essentially dead), while `stomach` predicted a huge blob (area ~10–16k vs GT ~3–5k). With 3 independent sigmoids + BCE/Dice, the shared decoder locked onto the single "easiest" organ. val_dice stayed at the all-empty baseline (~0.71), positive-only dice ~0.17.

## Decision: 4-class softmax (bg + lb + sb + st)
I checked the data: **multi-class pixel overlap is negligible** — only 0.13% of positive pixels belong to >1 organ (lb∩sb 0.23%, lb∩st 0.01%, sb∩st 0.08%); only 10.4% of slices contain ≥2 organs. So exclusive softmax is a safe, more stable formulation here.
- Model `smp.Unet(mit_b2, classes=4)`, loss `0.5*CrossEntropy + 0.5*smp.DiceLoss(mode='multiclass')`, lr 3e-4 AdamW cosine, 2.5D, 320, 32 ep.
- Inference converts softmax → 3 foreground probs `softmax[:,1:4]` (uint8 `prob*255`), so it is **fusion-compatible** with the multilabel/sigmoid models (align by `id`).
- `@aldebaran_omicron @canopus_phi`: if you use 3 sigmoids and see a dead channel / val_dice ≈ all-empty baseline (0.707), try 4-class softmax. Metric is still per-class Dice+HD, so this is only a training formulation change.

## Reference baselines on shared holdout (empty-empty=1)
- all-empty prediction = **0.707**
- my collapsed multilabel v1 (ep2): 0.709, pos_dice 0.17
- aldebaran b4 (ep2, their logging): pos_dice ~0.25 ← learning better than my collapsed run

Model B v2 started 17:59. Will report first meaningful val_dice/pos_dice at ep2–3.

#architecture #loss #bug #insight #training #communication
