set -e
cd /download_job_code/payload/sandbox/workspaces/capella_zeta
export OMP_NUM_THREADS=2
T10=$(cat specs_test_10.json)
python code/finalize.py --specs "$T10" --mode combine_test --weights "[1,1,1,1,1,1,0.5,0.5,0.5,0.5]" --out preds/ens10_test_probs.npy
python code/tune3d.py --mode submit --prob_file preds/ens10_test_probs.npy --ids_file preds/mitb3_f0_test_tta4_ids.csv --out_csv preds/sub_finalB.csv
echo "=== FINALB DONE ==="
