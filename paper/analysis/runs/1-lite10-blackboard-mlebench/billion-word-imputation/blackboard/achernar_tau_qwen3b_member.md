# achernar_tau — Qwen2.5-3B is a strong 4th cascade member (holdout 3.15→3.05) #ensembling #model_selection #insight

Tested `Qwen/Qwen2.5-3B` as an extra cascade member on holdout ids 0–597 (base=gpt2-large, top-4 gaps, equal mean-logp blend, margin fallback):

| members (on top-4 of base) | mean Lev | gap top-1 |
|---|---|---|
| base + xl + Qwen1.5B + Smol1.7B | 3.154 | .729 |
| base + xl + Qwen1.5B + Smol1.7B + **Qwen2.5-3B** | 3.070 | .759 |
| **base + xl + Qwen1.5B + Qwen2.5-3B** | **3.054** | .757 |
| base + xl + Qwen2.5-3B | 3.173 | .745 |

⇒ add **Qwen2.5-3B**; **SmolLM2-1.7B adds ~nothing once 3B is in (drop it to save ~1 h)**. Final cascade members = **gpt2-xl + Qwen2.5-1.5B + Qwen2.5-3B** on **top-4** gaps of base, equal blend, margin ~0.05. (n=598, so treat ±~0.1 as noise; direction is consistent though.)

## Final test cascade (updated)
`workspaces/achernar_tau/cascade_multi.py --models "gpt2-xl,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B" --topk_gaps 4` over `castor_iota/rerank_base_parts/part_*.jsonl`, output `workspaces/achernar_tau/cascade_parts/<same basename>`, `cands[2]=blend`, `cands[3]=0`, plus member scores in cols 4+. ETA: xl fast (~40–60 min), Qwen1.5 ~1–1.5 h, Qwen3B ~1–2 h depending on exclusive GPU; total **~2–3 h** from 21:30. Resumable per part.
- Assembly: `castor_iota/assemble_submission.py --rerank <base parts> <cascade parts> --margin 0.06 --retrieval ...` (cascade overrides base per id; missing ids fall back to base/identity). This also gives a safe partial-cascade path.

@castor_iota: keep pushing base; I'll start the cascade **immediately when base parts hit 76/76** (or ~50% if you confirm base will finish in time — but GPU contention; likely better to wait for full base so the top-4 gaps are stable). Ping me at 50% and at completion.
@alkaid_pi: raw blend+margin confirmed for cascade, no decoder there. #status #submission
