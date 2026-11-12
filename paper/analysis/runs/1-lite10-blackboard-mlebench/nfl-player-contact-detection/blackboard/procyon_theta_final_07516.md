# procyon_theta — final candidate 0.75158 (adds feat5_d2)

Read [[fomalhaut_omicron_pertype_0750.md]], [[procyon_theta_final_ping.md]].

Re-ran `ens_pertype.py` with **feat5_d2** auto-included:
- ground greedy: feat5_d2 + gruc + gru + feat5 -> **ground 0.7049**
- pair greedy: feat5×3 + gallr + l2 + feat5_d2 + gruc -> **pair 0.7697**
- **overall OOF <metric>0.7516<\metric>** (w5, thr 0.9829), test pos rate 1.62%.

**`workspaces/procyon_theta/sub_pertype.csv`** updated to this (best on board). @fomalhaut_omicron at your final write please use this file (or re-run `ens_pertype.py`); it includes feat5 + feat5_d2. I'm still not writing final_solution. If nothing lands by ~08:20 I'll do one write as warned.

#ensembling #submission #status #communication
