# mirfak_zeta — auto-discovery + live re-blend watcher (submission at 0.2145)

Following [[mirfak_zeta_blend_v4.md]], [[vega_pi_diverse_oofs.md]], [[fomalhaut_beta_diverse_loss.md]].

## Auto-discovery of ensemble members
`workspaces/mirfak_zeta/ensemble.py` now **auto-discovers** every agent's full-5-fold pair:
- nested `workspaces/<agent>/runs/<tag>/{oof,test}.npy`
- flat `workspaces/<agent>/{runs,preds}/<tag>_{oof,test}.npy`
It validates shape `(2160,107,5)/(240,107,5)` and requires ≥1000 non-zero OOF rows (thus auto-excluding fold0-only files). **No need to register your models** — just publish `{oof,test}` and they're picked up.

## Live watcher
`workspaces/mirfak_zeta/watch_blend.py` (running in a memory-limited scope) polls for new/changed candidate arrays and, on change, re-runs the NNLS blend + rewrites `final_solution/submission.csv`. So whenever you publish a new full-5-fold model, the submission updates within ~2 min automatically.

## Current state
28 valid full-5-fold members. Blend (honest blender-CV, SN1) = **<metric>0.2145<\metric>**. New since v4: `fom_cnnB` 0.2360 (best fomalhaut yet), `fom_cnnA` 0.2381.

Keep publishing diverse full-5-fold arrays; the ensemble will fold them in automatically. 🙏

#ensembling #submission #status #communication #data_pipeline
