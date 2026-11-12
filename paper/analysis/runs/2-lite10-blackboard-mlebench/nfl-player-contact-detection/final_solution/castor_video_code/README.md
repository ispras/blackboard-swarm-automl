# castor_alpha video pipeline (component of the final solution)

This folder documents the **video / helmet-crop ground-contact models** used in the swarm
final blend. The final `final_solution/submission.csv` is produced by the blender owner
(`rigel_delta`, see `../make_final3.py` / `../README.md`); these scripts generated the
`workspaces/castor_alpha/ground_{oof,test}_video_*.parquet` artifacts consumed by it.

## Pipeline
1. `extract_frames.py` — decode Sideline+Endzone videos at 10 Hz aligned to label/test
   `step` (`frame = round(290 + step*5.994)`, env `FRAME_OFFSET`). Output: JPEG frames.
2. `prep_helm_cache.py` — per-play helmet-box pickles from `*_baseline_helmets.csv`.
3. `build_crops.py` — per-(play,step,player) crops for ground rows (all positives + 4x
   sampled negatives): square crop around the helmet, size `GSCALE x helmet`, center
   shifted `GSHIFT x helmet` below the helmet (env `GSCALE`, `GSHIFT`). Output: uint8
   memmap `crops/<kind>_train.npy` + meta parquet. Pair crops use the union of two boxes.
4. `train_video.py` — 2-view CNN (shared encoder) + tabular branch; 5-fold GroupKFold by
   `game_key` (`folds.csv`); saves `<kind>_<arch>_f{0..4}.pt` and subset OOF.
5. `infer_video_stream.py` — streams full OOF/test rows from frames through the 5 fold
   models (test uses the fold average), writes contact_id-keyed probability parquet.
6. `ground_tab.py` — tabular ground baseline (tracking lags/rolling + helmet geometry +
   neighbour context), 95 features used in the video model's tab branch.

## Models used in the final ground blend (all trained on the shared folds)
| kind | arch | crop | full OOF AUC |
|---|---|---|---|
| ground | resnet18 | 3.6x / 0.9h | .9711 |
| ground | efficientnet_b0 | 3.6x / 0.9h | .9675 |
| ground | resnet34 | 3.6x / 0.9h | .9742 |
| ground | convnext_tiny | 3.6x / 0.9h | .9783 |
| ground | convnext_tiny.fb_in1k (seed2) | 3.6x / 0.9h | .9771 |
| ground | efficientnet_b3 | 3.6x / 0.9h | .9726 |
| **ground_wide** | **convnext_tiny.fb_in22k_ft_in1k** | **5.0x / 0.5h** | **.9801** |
| ground_tight | convnext_tiny.in12k_ft_in1k | 2.6x / 1.0h | .9749 |

Example commands (run via `systemd-run --scope` with a memory cap, `pyenv shell sis-black`):
```
FRAME_OFFSET=290 python extract_frames.py
python prep_helm_cache.py
GSCALE=5.0 GSHIFT=0.5 python build_crops.py ground_wide train 4
python train_video.py --kind ground_wide --epochs 3 --arch convnext_tiny.fb_in22k_ft_in1k --bs 128 --lr 2e-4
ARCH=convnext_tiny.fb_in22k_ft_in1k GSCALE=5.0 GSHIFT=0.5 \
  python infer_video_stream.py ground_wide train ground_oof_video_wide
ARCH=convnext_tiny.fb_in22k_ft_in1k GSCALE=5.0 GSHIFT=0.5 \
  python infer_video_stream.py ground_wide test ground_test_video_wide
```
Notes: test predictions are valid by construction (built directly from `data/test*` videos and
`test_baseline_helmets.csv`); no test labels were used. Thresholds are global rate-matched on
OOF (pair 1.20%, ground 4.60%).
