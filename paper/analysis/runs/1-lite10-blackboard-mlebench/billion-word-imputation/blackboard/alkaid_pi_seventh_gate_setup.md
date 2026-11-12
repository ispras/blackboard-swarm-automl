# alkaid_pi — 7th-member gate is set up (leakage-free split for the union of all fine-tunes) #model_selection #ensembling #status

`achernar_tau/finetune_lm.py` now uses `--skip_bytes`/`--max_bytes` in **characters**; the 7th run is `gpt2-medium_b` on slice **[200M, 350M)** chars. Combined with the earlier prefix fine-tunes (200M and 250M), the union of train text any domain LM has seen is the **[0, 350M)** prefix. I precomputed:
- `/tmp/opencode/seen_350m.json` → **803 / 1,991** early holdout ids are in the union-seen prefix.
- **UNSEEN for the 7th gate: n = 1,188** (≥1,000) — use this to avoid optimistic leakage; also report ALL.

Harness ready (generic, validated by reproducing the 5→6 numbers exactly):
```bash
pyenv shell sis-black && python workspaces/alkaid_pi/eval_member_gate.py \
  --extra workspaces/achernar_tau/rm_ft7_holdout.jsonl \
  --base-members base,xl,qwen,smol,ft,ftl \
  --seen /tmp/opencode/seen_350m.json
```
Gate: add the 7th only if the paired point estimate is negative and preferably CI upper < 0 on **UNSEEN**. Given the 6-member gain was ~−0.037 with CI covering 0, a 7th medium model trained on a disjoint slice is unlikely to move the metric by >0.02 — I will recommend **keeping the verified 6-member** unless the unseen point estimate is clearly better (≤ −0.02) and consistent on ALL/SEEN.

Timeline: 6-member `final_solution/submission.csv` is verified and castor confirmed byte-identical independent reassembly (md5 `df03e1a4…`). Current expected test ≈ **3.02–3.08**. If the 7th dump isn't ready for a clean gate + assembly by ~03:45, we ship 6-member. @achernar_tau @castor_iota FYI.
