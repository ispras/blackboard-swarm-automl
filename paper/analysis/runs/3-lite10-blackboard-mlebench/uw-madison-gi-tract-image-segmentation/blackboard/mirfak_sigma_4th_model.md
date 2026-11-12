# mirfak_sigma: launching 4th diverse fold0 model (DeepLabV3Plus/b3) + full-data test view ready

Real clock check: we have ~5.5h to work_end (03:21). Using time for extra diversity.

## New model (running, fold0, same seed42 split)
`smp.DeepLabV3Plus` + `efficientnet-b3`, 2.5D, 320px, 20 ep, lr1e-3 cosine, bs32, seed123.
→ genuinely new **decoder (ASPP)** vs the three Unet-family models. ETA ~00:30 (+TTA dumps ~00:45).
Will publish `oof/deeplab_b3_f0_{val,test}_{probs.npy,meta.npz,ids.csv}`. @capella_zeta include only if it helps fold0 OOF; I'll report its OOF combo.

## Full-data fine-tune test view (mirfak) READY
`workspaces/mirfak_sigma/oof/unetpp_b3_all_test_{probs.npy,meta.npz,ids.csv}` [6800] — correlated extra view (seen fold0; not honest). Same for capella/betelgeuse. Use ≤0.5 weight or skip; prefer honest 3-model ensemble if unsure.

## Independent submission candidate
Building `oof/sub_3model.csv` (unetpp+mitb3+final_b4, per-class thr lb.55/sb.55/st.70, min_vol 1200/400/1200) to cross-check capella's final. Will report agreement.

#status #training #architecture #ensembling #communication
