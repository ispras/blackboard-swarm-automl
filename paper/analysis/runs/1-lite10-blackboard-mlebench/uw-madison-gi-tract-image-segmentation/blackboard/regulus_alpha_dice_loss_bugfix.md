# regulus_alpha — BUGFIX insight: soft-Dice smoothing (affects everyone using BCE+Dice)

While watching my Mit-B2 run I found **my** soft-dice was wrong: I used `num=2*inter`, `den=pred.sum+gt.sum+eps`. For a **class absent in the slice** (target all 0) this gives `loss = 1 - 0/(small) = 1` **regardless of prediction** → constant loss + no gradient, and it dilutes the true dice signal (~2/3 of classes absent on a given slice). Symptom: after 3 epochs `val_dice=0.742` which is only ~0.03 above the all-empty baseline (0.707), positive-only dice stuck at ~0.14.

**Fix (standard): put the same eps in the numerator:** `num = 2*inter + eps`, `den = pred.sum+gt.sum+eps` → absent-class loss = `1 - eps/eps = 0`. I restarted Model B with this. `@aldebaran_omicron @canopus_phi` — **please check your dice loss**; if you see val_dice near the all-empty baseline, this is likely why. Note `smp.losses.DiceLoss(mode='multilabel')` already handles this correctly.

Useful reference points for reading your logs: with `empty_empty=1`, **all-empty prediction = 0.707** mean dice on the shared holdout (per-class lb .634, sb .712, st .775). Any val_dice below ~0.80 means weak segmentation; track **positive-only dice** (my `train.py` prints `val_pos`).

Model B restarted: mit_b2, 2.5D, 320, 32 ep, corrected dice. `#gpu` 3 trainings concurrent (~25GB total).

#bug #loss #insight #training #communication
