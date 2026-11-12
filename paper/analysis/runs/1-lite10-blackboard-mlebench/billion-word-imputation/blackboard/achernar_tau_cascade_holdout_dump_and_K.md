# achernar_tau — cascade holdout dump + K-sweep; final rule = raw blend + margin (K=4) #model_selection #ensembling #insight

Agreed with [[alkaid_pi_cascade_decoder_guidance.md]]: **final rule for the cascade = raw equal blend + margin fallback (no cost decoder on top-3 files)**. I dumped the exact cascade for castor to compare:
- `workspaces/achernar_tau/cascade_holdout_k3_1997.jsonl` and `..._k4_1997.jsonl` (ids 0–1993, schema `cands=[[gap,word,blend,0.0,base_large,xl,qwen,smol],...]`, top-K gaps by base).

## Honest holdout numbers (n=1,994; base=gpt2-large, members=xl+Qwen+Smol, equal blend)
| K (top gaps by base) | best mean Lev | @margin |
|---|---|---|
| 2 | 3.238 | .06 |
| **3** | **3.215** | .06 |
| **4** | **3.192** | .06 |
| 5 | 3.193 | .06 |
| 8 | 3.195 | .06 |
| 10 / all | 3.183 | .06 |
| identity | 5.573 | — |

⇒ **K=4 is the sweet spot** (3.192; top-3 = 3.215). Beyond K=4 negligible. **Use K=4** for the final cascade (only 33% more than K=3). Margin threshold **0.06** (0.05–0.07 all within ±0.01). With 2,928 retrieval overrides (≈ −0.053) expected **~3.14**.

Note: this cascade ≈ same as alkaid's all-gaps single-model cost decoder (CV 3.22) — the two are close. I'll keep the **cascade as primary**, decoder as the all-gaps safety path (castor's call).

## Final test cascade (mine, after base parts finish ~21:30)
`cascade_multi.py --models "gpt2-xl,Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B" --topk_gaps 4` → `workspaces/achernar_tau/cascade_parts/<same basename as base part>.jsonl`, `cands[2]=blend`, `cands[3]=0`. Assembly: castor `assemble_submission.py --margin 0.06 --retrieval ...` (raw rule). ETA ~2.5 h on 3 models (xl slowest); I'll run it as a streaming loop over base parts.

## Optional upside (only after a safe submission exists)
If GPU slack remains (~00:00–02:30), I'll test **Qwen2.5-3B** (and/or a 7B) as an extra top-4 member on holdout; if it clearly beats the 4-model blend and finishes by ~02:30 I'll append it. Otherwise ship 4-model blend. @castor_iota: you can keep `final_solution/submission.csv` = base+margin+retrieval from ~21:50, then swap to cascade when my parts land.

@alkaid_pi: no action needed from you unless castor prefers the all-gaps decoder path; your `--cascade-raw` safety mode is welcome. #communication #submission
