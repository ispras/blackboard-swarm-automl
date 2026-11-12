# analysis/ — shared read-only artifacts (owner: alderamin_xi)
- `holdout_20k.csv` — canonical validation holdout (copy of rigel_mu's `workspaces/rigel_mu/holdout_20k.csv`). Use this for ALL comparable val numbers.
- `val_holdout.csv` — older 40k split (img_id md5 hash); prefer holdout_20k.
- `eval_lev.py` — mean Levenshtein scorer:
  `pyenv shell sis-black && python analysis/eval_lev.py --pred PRED.csv --truth analysis/holdout_20k.csv`
  PRED.csv needs `image_id,InChI`.
Do not edit files here without telling the board; add your own under your workspace.
