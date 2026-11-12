# Near-final submission written (partial rigel 261k covered); projected ~8-9; will refresh at full coverage
#submission #status

`final_solution/submission.csv` now uses the **final rule** (canon(rigel) + 1m high-confidence override, T=0.85) with the current partial `rigel_test_all.csv` (261,505/484,838 rows at 16:51). Rows not yet covered by rigel fall back to 1m (113,709 rows). Format validated: 484,838 rows, ids exact, all start `InChI=1S/`, placeholders 1,598.
- Full holdout of this exact rule: **7.76** (canon(rigel)+1m≥0.85); rigel-alone 8.58. So even the partial submission is already ≪ the TTA/gate safety nets.
- When `rigel_test_all.csv` completes (~17:25) I rerun `finalize_rigel.py` (same T=0.85) -> final ~7.8, then write README+code and ping for independent validation.
**rigel_mu**: all-test is the only remaining critical job; finish it. altair: nothing needed. Tags: #submission #status
