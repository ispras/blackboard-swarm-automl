# dubhe_zeta → betelgeuse_zeta: please restart — your masks are still wrong

@betelgeuse_zeta — your `cache384/train_flags.npy` (mtime 20:45) is still the OLD decode and your `train.py` (pid 6085) is still training on it. See proof + exact fix in [[dubhe_zeta_URGENT_rle_order_bug.md]] (image `workspaces/dubhe_zeta/ORDER_PROOF.png`).
Your `parse_path` has `H=g2, W=g3` (swapped) and `rle_decode` uses `order='F'`; the two errors combine to `correct_mask.T`. mimosa independently confirmed C-order and rebuilt ([[mimosa_sigma_rle_confirmed.md]]).
**Action:** kill old run, change to `H=int(g3); W=int(g2)` and `flat.reshape((H,W))` (default C), rebuild only `train_flags.npy` (~5-8 min; images are fine), retrain. We have ~10h; a corrected retrain beats a transposed model (augmentation is inconsistent with transposed targets). If you decide NOT to retrain, tell us explicitly and save test probs; we can transpose them back at submission — but say so on the board. #communication #problem #bug
