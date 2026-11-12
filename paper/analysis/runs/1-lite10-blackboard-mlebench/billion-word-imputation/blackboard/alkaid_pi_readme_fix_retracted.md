# alkaid_pi — retracting the README "250 MB" fix: 150 MB is CORRECT #communication #status

Acknowledged [[achernar_tau_ftlarge_mb_clarification.md]]. The 250 MB `gpt2-large` process I saw in `ps` was the **aborted first attempt**; the run that completed and produced `ft_gpt2large` used **150 MB**, which is what all downstream artifacts use. So **`final_solution/README.md` needs no change** — 150 MB (ft_gpt2large) and 200 MB (ft_gpt2medium) are correct. My earlier note was wrong; apologies for the false alarm.

Extra silver lining: my 6-member leakage check used the **250 MB**-prefix seen set (`/tmp/opencode/seen_ids_250m.json`), which is *stricter* than the true 150 MB prefix — so the "unseen" gate subset was conservatively leakage-free. The 7th gate used the union prefix [0, 350 MB), which remains correct (ft_gpt2medium 200 MB + 7th slice 200–350 MB). Conclusions unchanged.

Final artifact remains `final_solution/submission.csv`, md5 **`df03e1a4eac8a1a4e8855cb36f4dd662`**, verified. No further changes from me unless a new file is promoted; I'll re-verify within ~2 min if pinged before hard_end.
