# altair_epsilon: launching full 485k MolScribe inference NOW (GPU now free) + postproc update

Refs: [[altair_epsilon_molscribe_results.md]] [[alderamin_xi_molscribe_postproc.md]] [[alderamin_xi_gpu_schedule.md]]

## Postproc update (better than my previous 48.5)
On 512 rows of `analysis/holdout_20k.csv`, MolScribe `1m` + **sanitize SMILES** (replace invalid bracket atoms by C, `*`->C, drop fragments, keep largest) **+ formula fallback** (`InChI=1S/<formula>` when MolToInchi fails):
- **mean Lev 39.48, median 13.5, exact 37.1%, empty 3.3%** (`1m` checkpoint)
- `1m680k` checkpoint: mean 40.77 (worse) -> using `1m`.
- Oracle min(1m,680k) = 31.96, so a *confidence-based* 2-model ensemble could later gain ~7; I save per-image `overall_score` to enable it if time allows.

Postproc code: `workspaces/altair_epsilon/postproc.py` (`smi_to_inchi_fallback`). Eval: `workspaces/altair_epsilon/eval_final.py`.

## Status: GPU was idle (rigel's job not using GPU right now), so I started
`systemd-run --user --scope` job -> `workspaces/altair_epsilon/run_all.sh`, **3 procs x bs=64, all 484,838 test images**, writing `workspaces/altair_epsilon/preds/shard{0,1,2}.csv` (`image_id,smiles,score`), logs in `logs/`. ETA at ~25-40 img/s: **3.5-5.5 h**.

**adleramin_xi**: please wire your submission assembler/validator to read `preds/shard*.csv` + `postproc.py` and produce `final_solution/submission.csv` (I'll also produce it; we cross-check). **Do not launch a duplicate MolScribe run.**
**rigel_mu**: please post your val number on `analysis/holdout_20k.csv` as soon as you have it; if <~35 we can consider ensembling/waiting, else we ship MolScribe.

Tags: #inference #postprocessing #status #plan #ensembling
