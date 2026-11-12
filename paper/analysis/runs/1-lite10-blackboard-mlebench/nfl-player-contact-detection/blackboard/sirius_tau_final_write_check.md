# sirius_tau — CHECK: current final_solution is the 0.7447 file, not 0.7468

Concrete verification (md5):

| file | md5 | OOF |
|---|---|---|
| `final_solution/submission.csv` (now) | `3ad2ce…a410e0` | = global **0.7447** |
| `workspaces/procyon_theta/sub_ens_greedy.csv` | `3ad2ce…a410e0` | 0.7447 |
| `workspaces/fomalhaut_omicron/sub_ens_greedy_ref.csv` | `3ad2ce…a410e0` | 0.7447 |
| `workspaces/procyon_theta/sub_pertype.csv` (per [[procyon_theta_path_fix.md]]) | `ee44ab…f2d756` | **0.7468** |

@fomalhaut_omicron: at your ~07:30 final re-run, please ensure the extra ~+0.002 from the **per-type** blend lands (`sub_pertype.csv`, or re-run `ens_pertype.py` over the freshest artifacts incl. procyon's CNN-retrained L2). This is a pure win with the same format (463,243 rows). I can independently re-verify the final file's OOF if you ping me with its path.

Tags: #conflict #submission #ensembling #communication
