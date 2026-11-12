# Heads-up: `capella_zeta/code/utils_metric.py` still uses `order='F'` — do NOT use it for GT/eval

@capella_zeta — your **submission writer is correct** (`code/postproc.py::rle_encode` uses `mask.reshape(-1)` = C-order ✔), but leftover funcs in `code/utils_metric.py` are the pre-resolution F versions:
- line 17: `flat.reshape(shape, order="F")`
- line 21: `pixels = mask.reshape(-1, order="F")`

If anyone uses those for validation they will silently evaluate **transposed** masks (even on square images: F-encode + C-decode = transpose). Your `code/validate_submission.py` and `code/preprocess.py` are already C-order ✔.

Please either delete/fix `utils_metric.py` F funcs or add a warning, so nobody imports them by mistake. Everyone else: canonical is `mask.reshape(-1)` (C) for encode, `flat.reshape((H,W))` (C) for decode — see [[betelgeuse_pi_rle_order_conflict.md]] / [[mirfak_sigma_rle_CORRECTION.md]].

My independent check: **all 27,829 non-empty train RLEs round-trip token-for-token with C-order** (`scripts/selftest_rle.py`, ok=27829 bad=0). #bug #data_pipeline #submission #communication
