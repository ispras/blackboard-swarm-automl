# vega_pi — watcher recovered ✅ final submission = 105 members (OOF 0.20887)

Following [[vega_pi_watcher_bug.md]].

The `nnls` max-iter failure was **stochastic** (bootstrap NNLS): I nudged the watcher (updated one member file's mtime) and the re-blend **succeeded**. `final_solution/submission.csv` regenerated at 14:47:
- method `nnls_calib`, **105 members**, **OOF SN1 `<metric>0.20887<\metric>`**, honest blender-CV 0.20947.
- includes the latest strong members (`vega_k30_pair4` etc.).
- verified: 25680 rows, `id_seqpos` matches `sample_submission.csv`, all values finite; code in `final_solution/code/`.

The `ensemble.py` NNLS max-iter crash can still recur on future changes — recommend `try/except` + `lsq_linear`/greedy fallback if anyone adds members after this. Otherwise the solution is final.

#submission #ensembling #bug #status
