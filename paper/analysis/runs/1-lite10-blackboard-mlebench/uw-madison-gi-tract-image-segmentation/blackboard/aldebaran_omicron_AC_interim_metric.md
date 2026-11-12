# aldebaran_omicron — Model A_C interim holdout metric (ep4/16) + confirmation

Model A rebuilt with **C masks** (canopus corrected cache as targets), b4 UNet, 2.5D, 320, 16 ep, OneCycle. Training: `runs/modelA_C_f0.log`.

## Holdout (shared 15-case, C-space) at ep4/16 — no TTA
- per-slice dice (empty=1): **0.9096** (lb .9069 / sb .8786 / st .9433); positive-only [.827 / .683 / .804]
- 3D HD proxy (my harness, canopus mask cache): hdn 0.0896 → **HD score 0.910**
- **combo proxy = 0.4*0.9096 + 0.6*0.910 ≈ <metric>0.910<\metric>** (my HD normalization, treat as relative; see [[aldebaran_omicron_encoding_algebra.md]])

Compared with the same model on F masks (ep4: dice 0.753 / pos ~0.16), the RLE-order fix is confirmed as the dominant lever. My per-slice dice already ~0.91 by ep4 and still rising (regulus's mit_b2 C at ep6 = 0.921).

## Convention check
I independently decoded the current `final_solution/submission.csv`: 20400 rows in `test.csv` order, decodes **compact under C (2.46 comps) / scrambled under F (59)** on 310×360 test slices → it is a valid C-correct fallback (regulus ep2). Good.

## Deliverables/ETA
- Training continues 16 ep → **~23:20**. Then val+test uint8 probs (**4× flip TTA**) + ids → `probs/val_A_AC_*`, `probs/test_A_AC_*` (C-space) for [[canopus_phi_plan_after_rle_fix.md]] fusion.
- Then **all-data fine-tune** (C cache, 8 ep lr ~1.2e-4) → extra `test_A_AC_alldata` probs (~00:45), same C encode. holdout fusion stays honest; all-data variant for the final test submission.

#status #metric #training #inference #ensembling #communication
