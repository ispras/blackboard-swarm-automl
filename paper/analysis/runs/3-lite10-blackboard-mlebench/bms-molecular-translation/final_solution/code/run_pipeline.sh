#!/bin/bash
# End-to-end (paths relative to repo root). Requires GPU; run under systemd-run --scope with a memory cap.
set -e
ROOT=${ROOT:-/download_job_code/payload/sandbox}
# 1) MolScribe inference on all test images (shard over 4 processes)
for s in 0 1 2 3; do
  python code/molscribe_infer.py --ids $ROOT/workspaces/altair_epsilon/test_ids.csv --imgroot $ROOT/data/test \
    --out molscribe/shard$s.csv --bs 128 --shard $s --nshard 4 --conf &
done; wait
# 2) SMILES->InChI postprocess (see postproc.py) -> molscribe/preds1m_inchi.csv (image_id,inchi,valid,score)
# 3) build rigel ids = rows with (not valid) OR (score < 0.7)
# 4) our model inference on those ids (fast KV-cache decoder)
python code/rigel_infer_fast.py --ckpt ck/v1_best.pt --ids rigel_infer_ids.csv --imgroot $ROOT/data/test --out rigel_test.csv
# 5) final assembly: canon(rigel) with 1m override when 1m valid & score>=0.85
python code/finalize_rigel.py --inchi1 molscribe/preds1m_inchi.csv --rigel rigel_test.csv \
  --score_thr 0.85 --out final_solution/submission.csv
python code/validate_submission.py --sub final_solution/submission.csv
