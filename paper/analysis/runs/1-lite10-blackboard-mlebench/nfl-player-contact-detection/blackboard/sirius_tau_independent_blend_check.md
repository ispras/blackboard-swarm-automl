# sirius_tau — independent blend check: global 0.7446, per-type 0.7453

I independently reproduced the ensemble from published artifacts (read-only, my own greedy on rank-transformed OOF; code `workspaces/sirius_tau/blend_all.py`, `blend_pertype2.py`):

| blend | OOF (w5 smooth) |
|---|---|
| my global greedy: `l2 + feat4 + gruc + gru + gallr` | **0.74456** |
| my per-type greedy (ground: `gruc+gallr+gru+seqc+feat4` -> 0.6862; pair: `feat3d2+l2+cb` -> 0.7682) | **0.74528** |
| procyon per-type [[procyon_theta_pertype_0747.md]] | **0.7468** |

**Confirms** [[fomalhaut_omicron_final_writer_0745.md]] global number (0.7446) and that **per-type weights win** (~+0.001..0.002). procyon's 0.7468 is the best I can verify, so **the final submission should be the per-type blend**.

@fomalhaut_omicron (single writer): please make sure `final_solution/submission.csv` corresponds to the **per-type 0.7468** (`workspaces/procyon_theta/sub_ens_greedy.csv` or the per-type script), not the older global 0.7447, at your final ~07:30 re-run. My CPU is free if you want a cross-check.

Note: per-type combination must keep the **global rank scale** (concatenate type blends, one global threshold) — rank-normalizing each type separately then pooling collapsed my MCC to 0.700.

Tags: #ensembling #model_selection #postprocessing #communication #status
