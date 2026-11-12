# canopus_pi solution code (LightGBM full-data + corrupt model)

Pipeline:
1. `prep.py` — chunked CSV -> parquet features (cleaning: 2.5<=fare<=500, coords bbox lon[-75.5,-72.5] lat[40,42], pax 0-8; features: coords/deltas, haversine, manhattan, bearing, 29deg-rotated coords, airport/center dists, datetime; carries `orig_idx`).
   - `python prep.py train 1.0 42` (full) and `python prep.py holdout` (shared holdout `orig_idx%100==0`).
2. `train_lgb4.py` — full-data LightGBM with integer multi-resolution spatial target encoding (0.01deg pickup/dropoff cells + OD route, 0.001deg pickup), 5-fold OOF TE. Produced `g54m` (511 leaves, lr .06) -> valid holdout RMSE 3.33317.
   - `python train_lgb4.py --prepdir <prep_full> --maxrows 60000000 --lr 0.06 --leaves 511 --tag g54m --save --threads 20`
3. `predict4.py` — rebuilds the training TE maps and predicts `data/test.csv` (test.csv order).
4. `corrupt2.py --build` + `corrupt_train.py` — corrupt-coordinate features (per-coordinate validity, transpose recovery, valid-endpoint TE) and LightGBM; 5-fold CV RMSE 12.6537.

No external datasets or pretrained models. Libraries: numpy, pandas, pyarrow, lightgbm.
