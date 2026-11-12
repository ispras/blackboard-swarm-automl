set -e
cd /download_job_code/payload/sandbox/workspaces/capella_zeta
export OMP_NUM_THREADS=2
F0=$(cat specs_fold0_6.json)
T9=$(cat specs_test_9.json)
echo "=== combine val 6 ==="
python code/finalize.py --specs "$F0" --mode combine_val --weights "[1,1,1,1,1,1]" --out preds/ens6f_val_probs.npy
echo "=== tune3d 6 ==="
python code/tune3d.py --mode tune --prob_file preds/ens6f_val_probs.npy --ids_file preds/mitb3_f0_val_tta4_ids.csv
echo "=== combine test 9 ==="
python code/finalize.py --specs "$T9" --mode combine_test --weights "[1,1,1,1,1,1,0.5,0.5,0.5]" --out preds/ens9_test_probs.npy
echo "=== submit 9 ==="
python code/tune3d.py --mode submit --prob_file preds/ens9_test_probs.npy --ids_file preds/mitb3_f0_test_tta4_ids.csv --out_csv preds/sub_finalA.csv
echo "=== FINALA DONE ==="
