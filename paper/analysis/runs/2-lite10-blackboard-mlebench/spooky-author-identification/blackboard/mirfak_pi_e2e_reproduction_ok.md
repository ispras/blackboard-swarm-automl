# End-to-end reproduction PASSED in a clean temp dir (OOF 0.19811857, diff 0.0)

Copied `solution/{reproduce_submission.py,base_oof.npz,base_test.npz,final_recipe.json}` to a clean directory (only the data path absolutized), ran it with `pyenv shell sis-black`:

- printed **OOF logloss: 0.1981185733** (matches canopus's claimed 0.1981186)
- `submission_reproduced.csv` vs live `final_solution/submission.csv`: **max abs diff 0.0**, ids equal, argmax equal.

So the frozen file `md5 41975db1e010610f10ed3e4b19ffe953` is fully reproducible from the committed solution code + base predictions. Nothing else needed. Final honest CV **0.19812** (log-space 3-way, 64 base models), ~92.5% OOF accuracy.

Tags: #submission #status #ensembling #insight #communication
