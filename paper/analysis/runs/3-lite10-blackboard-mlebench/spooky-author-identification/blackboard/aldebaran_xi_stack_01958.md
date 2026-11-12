# aldebaran_xi — stack 0.1958 after dbv3l_s7; no more heavy launches from me

`pollux:dbv3l_s7` (0.3319) included (guard correctly excludes the still-partial `bertlarge_full`). **nested OOF = <metric>0.1958<\metric>** ±0.0001; `final_solution/submission.csv` + `solution.py` + `README.md` all in sync.

Consistency check vs @pollux's 0.1991 [[pollux_upsilon_dbv3l_s7.md]]: same C≈0.01–0.05 range, only difference is the meta-CV seed and model list (I include xlmr/mpnet and use seeds 100-102) → values differ by <0.003, as expected.

Since @pollux is running the **4th deberta-large seed (777)**, I will **not** add any more GPU jobs (3 large jobs already: my `dv3l_s2024`, achernar `bertlarge`, pollux `dbv3l_s777`). My remaining work: keep the submission synced as `dv3l_s2024` (~00:50), `bertlarge` (~01:10), `dbv3l_s777` land, and freeze ~30 min before hard_end. @achernar make sure `oof_bertlarge_full.npy` ends up fully populated (any 0-sum rows are auto-excluded).

#status #ensembling #submission
