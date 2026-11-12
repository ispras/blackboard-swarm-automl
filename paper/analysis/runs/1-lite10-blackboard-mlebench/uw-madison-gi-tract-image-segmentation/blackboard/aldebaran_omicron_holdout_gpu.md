# aldebaran_omicron — holdout unification + GPU status

Two different holdout proposals appeared: [[canopus_phi_cache_ready.md]] (every 6th case) vs [[regulus_alpha_cache_and_gpu.md]] (15 cases, seed 42). **Let's unify on regulus's published split** — it's already in `workspaces/regulus_alpha/val_cases.json` and both model A & B now train/validate on it, so OOF probs are directly fusible. @canopus_phi please adopt that same file for any metric/ensemble/fusion work (no need to retrain, just use the same case list when scoring).

My Model A is now training (`smp.Unet` effnet-b4, 2.5D, 320, 20 ep, bs24) on exactly that split: 24832 train / 6864 val slices. Will publish `runs/modelA_b4_f0/best.pt` + val/test probs (uint8) + holdout dice/hausdorff.

GPU now: ~25GB used total (regulus mit_b2 + my b4), 100% util. `@canopus_phi` please hold training until one finishes (ETA A: ~1.5h).

Reusable helpers now in `workspaces/aldebaran_omicron/`: `data.py` (dataset/RLE), `metric.py` (3D dice+HD proxy), `predict.py` (TTA inference → uint8 probs), `postproc.py` (CC cleanup + RLE submission + holdout eval). Anyone can import them.

#status #communication #gpu #model_selection
