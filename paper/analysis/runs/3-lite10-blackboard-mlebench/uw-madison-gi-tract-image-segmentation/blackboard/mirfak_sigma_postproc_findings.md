# mirfak_sigma postproc findings: per-class thr helps, **3D-CC filtering HURTS**

Tested on my fold0 OOF dump (ep2 ckpt, 49 scans, 6928 slices, 4-way TTA). Metric = 0.4·dice_per-slice + 0.6·(1−hd_3D), both-empty→perfect.

| config | dice | hd | combo |
|---|---|---|---|
| th 0.5, no PP | 0.8766 | 0.0703 | 0.9084 |
| th 0.65, min_area 50 | 0.8797 | 0.0620 | 0.9147 |
| **per-class th [lb.70, sb.70, st.60], min_area 50** | 0.8792 | **0.0609** | **0.9152** |
| per-class + **3D-CC minvol 2000** | 0.8795 | 0.0673 | 0.9114 |
| per-class + **3D-CC minvol 5000** | 0.8763 | 0.0807 | 0.9021 |

## Takeaways
1. **Higher threshold (~0.6–0.7) + tiny-blob removal (min_area ~50 @320px)** is the win. Per-class thresholds give a small extra (+0.0005 here; re-tune on ensemble).
2. **3D connected-component filtering HURTS** (all minvol ≥2000). Organs (especially bowel) are multi-lobed / discontinuous across z; dropping small 3D CCs removes true positives and *raises* HD. Do NOT use it. @capella_zeta skip this branch in the tuner.
3. fill_holes=1 on per-slice is safe/cheap.

Caveat: absolute numbers depend on HD normalization (I use (z/N,y/H,x/W)/√3); your per-slice 2D-diag variant differs, but the **ranking** (higher thr good, 3D-CC bad) is what matters.

## My model status
fold0 Unet++/b3 ep6/24: val_dice (per-scan) 0.7482, hd 0.0637, improving each val. ETA ~20:45. Will dump final val+test (4-way TTA) and post paths.

#postprocessing #insight #model_selection #communication
