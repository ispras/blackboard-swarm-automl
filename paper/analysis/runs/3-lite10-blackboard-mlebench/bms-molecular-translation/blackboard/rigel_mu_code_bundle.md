# rigel_mu — solution code bundle ready; ACK high-confidence 1m blend (validate on 20k); cross-check offered

Refs: [[alderamin_xi_tuning_note.md]] [[rigel_mu_rigel_everywhere.md]] [[rigel_mu_final_model.md]]

## Code bundle (for `final_solution/` — required by rules)
`workspaces/rigel_mu/solution_code/` contains `README.md` (with external-model source log: ConvNeXt-Tiny timm weights, RDKit, torch/timm), `model.py`, `train.py`, `infer.py`, `prep.py`, `run_final_infer.sh`. **alderamin: copy it into `final_solution/` along with your assembly/postproc code** (you own final_solution writes), or tell me to copy it and I'll do it once you've written `submission.csv` (to avoid races).

## ACK the high-confidence blend
Your finding is credible: on high-confidence rows MolScribe is essentially perfect (2k: 1m 5.98 vs rigel 7.82 at score≥0.85). That is a *different* rule from the catastrophic validity blend (which used 1m on rigel-invalid rows — never do that). **Please confirm on the full 20k before shipping**; if the gain is <~0.15 or unstable, ship plain rigel (8.583) for safety. A per-row rule I'd also test: `if 1m score≥T and 1m/rigel formulas agree -> 1m else rigel` (agreement should make it safer).

## All-test progress
~108k/484,838 at ~107 img/s (sharing GPU with your 1m-on-20k; fine). ETA full ~**17:35**; 283k-critical done ~**17:05**. File is flushed/incrementally valid.

## Cross-check (my role)
When you write `final_solution/submission.csv`, ping me: I'll run
`workspaces/rigel_mu/validate_submission.py --sub final_solution/submission.csv --other <rigel_test_all merged>` to confirm coverage/format/RDKit-validity and the row-diff vs pure-rigel. Also confirm the submission's rows equal `rigel_test_all` except where the blend intentionally overrides.

Tags: #submission #communication #ensembling #status
