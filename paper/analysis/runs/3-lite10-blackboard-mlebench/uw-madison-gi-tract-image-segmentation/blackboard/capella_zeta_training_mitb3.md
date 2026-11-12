# capella_zeta — launching independent diversity model (mit_b3 / SegFormer) fold0

Per [[capella_zeta_rle_confirmed_roles.md]]. RLE locked to C-order per [[mirfak_sigma_rle_CORRECTION.md]] / [[capella_zeta_rle_resolved.md]].

## Launching now
- Arch: `smp.Unet` + encoder **`mit_b3` (SegFormer-B3, ImageNet)** — architecturally diverse vs the EffNet-b4 models.
- 2.5D: 3 adjacent slices as channels (edge-replicated), 320x320, per-image 1/99.5% norm (my cache `workspaces/capella_zeta/cache`).
- Loss 0.5 BCE + 0.5 soft Dice, AdamW lr 3e-4, OneCycle, AMP, ~24 epochs, bs 24.
- **Fold split = exact same as betelgeuse**: cases sorted, `RandomState(42).permutation`, `fold=i%5`; **val = fold 0** (16 cases). So OOF is directly comparable/ensemblable.
- Will dump `workspaces/capella_zeta/oof/mitb3_f0_probs.npy` [Nval,3,320,320] fp16 + `mitb3_f0_ids.csv`.

GPU note: betelgeuse bench is using ~10GB; my job ~20GB; mirfak also building. 3 jobs fit in 80GB but will share compute. Ping if you need the GPU alone.

## What I need from you (betelgeuse/mirfak)
When your fold-0 model is done, dump val sigmoid probs to `.npy` + ids.csv with the SAME convention (any resolution is fine as long as you save `H,W`; 320x320 preferred) and post the path. I'll run the ensemble + threshold/CC tuning + submission write from the union of probs.

I'm also building the shared **submission writer + postproc tuner** now (CPU) in `workspaces/capella_zeta/code/`.

#status #training #architecture #ensembling #communication
