#!/usr/bin/env bash
# Reproduction sketch for the swarm ensemble submission.
# Run from anywhere; paths are relative to the sandbox root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ME="$ROOT/workspaces/bellatrix_omicron"

# 1) Feature cache for the 50-feature family (writes X2_train.npy, X2_test.npy, feature_meta2.json)
python "$ROOT/final_solution/preprocess2.py"

# 2) 5-fold + all-data models of the 50-feature family (fold = breath_id % 5)
#    Each command writes oof_<tag>_f<k>.npy / test_<tag>_f<k>.npy in the workspace.
for k in 0 1 2 3 4; do
  python "$ROOT/final_solution/train_lstm3.py" --fold $k --tag physfull --epochs 120 \
    --hidden 512 --layers 4 --head_layers 2 --head_hidden 256 --batch 256 --lr 2e-3 \
    --dropout 0.15 --wd 3e-4 --exp_weight 0.1 --seed 2024
done
for s in 777 999 555; do
  python "$ROOT/final_solution/train_lstm3.py" --fold 0 --tag physall_$s --all_data --save_last \
    --epochs 100 --hidden 512 --layers 4 --head_layers 2 --head_hidden 256 --batch 256 --lr 2e-3 \
    --dropout 0.15 --wd 3e-4 --exp_weight 0.1 --seed $s
done

# 3) Merge the row-aligned OOF/test predictions of the three members
#    (c34 / ald artifacts are produced by peer agents in workspaces/canopus_omega and
#     workspaces/aldebaran_eta; see final_solution/README.md), fit simplex MAE weights
#    with leave-one-fold-out CV, and write final_solution/submission.csv.
python "$ROOT/final_solution/blend_final.py" \
  --out "$ROOT/final_solution/submission.csv" \
  --final_dir "$ROOT/final_solution"
