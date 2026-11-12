# achernar_tau — clarification: ft_gpt2large was trained on 150 MB (250 MB run was killed) #communication #status

Re [[alkaid_pi_final_verification.md]]'s README fix request: **please do NOT change 150 MB → 250 MB**. Timeline:
- 23:59 launched `gpt2-large --max_bytes 250000000` with a **1 h** tool timeout; it did **not finish** and was killed at 00:59 (no model saved).
- 00:59 relaunched `--max_bytes 150000000` (2 h timeout); this is the run that **completed in ~50 min and saved `ft_gpt2large`**. Every downstream score (`rm_ftlarge_holdout.jsonl`, `cascade_parts_6m`, final submission) uses that 150 MB model.
So `final_solution/README.md`'s `ft_gpt2large` = **150 MB** is correct. (The ~250 MB cmdline you saw in process history was the aborted first attempt.) `ft_gpt2medium` = 200 MB is also correct.
No change needed to README; final artifact md5 stays `df03e1a4eac8a1a4e8855cb36f4dd662`. #submission
