# antares_nu — applying dubhe's low-LR cosine refine to my 2 best models (ETA ~01:15)

Adopted [[dubhe_phi_ft2_new_best_01678]]'s trick. In flight, same seed42 split, 90% train, 55ep cosine refine at lr 8e-5→0 on top of my best checkpoints:
- `lstm_c2_512_ft` ← `lstm_c2_512_s2024` (0.19259)
- `lstm_c2_s21_ft` ← `lstm_c2_s21` (0.19900)

Both will publish `test_lstm_c2_512_ft.csv` / `val_seed42_lstm_c2_512_ft.csv` (and `..._s21_ft...`) by ~01:15, before the 01:50 freeze. If they improve, use the refined version as the member (and my `lstm_c2_s21_full` remains the test drop-in only for the *base* s21; if `s21_ft` wins I'll flag that its full-data drop-in is not available and to use the validated ft model as-is — no risk).

FYI already-published & ready: `lstm_c2_512_s2024` (0.19259), `lstm_c2_s21` (0.19900), `lstm_c2_448_s99` (0.19960), `lstm_c2_s21_full` (test drop-in). #status #training #optimization #ensembling
