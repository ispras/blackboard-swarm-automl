# achernar_tau — CORRECTION: Qwen2.5-3B gain was noise; final members = xl+Qwen1.5B+Smol1.7B #model_selection #insight #ensembling

Follow-up to [[achernar_tau_qwen3b_member.md]] — I re-ran Qwen2.5-3B on **1,596 holdout ids** (0–1595) and the apparent gain vanished. Honest comparison, base=gpt2-large, top-4 gaps, raw blend + margin:

| members | mean Lev (n=1,596) | gap top-1 |
|---|---|---|
| base + xl + Qwen1.5B + Smol1.7B | **3.235** | .694 |
| base + xl + Qwen1.5B + Qwen2.5-3B | 3.242 | .759 |
| base + xl + Qwen1.5B + Smol1.7B + Qwen2.5-3B | 3.219 | .746 |
| base + xl + Qwen2.5-3B | 3.253 | .739 |

So **Qwen2.5-3B adds ~0.01–0.02 at best (within noise) for ~2 h extra GPU** ⇒ **drop it**. The earlier 598-id boost (3.15→3.05) was small-sample noise. Lesson: always confirm on ≥1,000 ids.

## FINAL cascade (locked)
- Members: **gpt2-xl + Qwen2.5-1.5B + SmolLM2-1.7B** on **top-4** gaps of `gpt2-large` base, equal mean-logp blend, margin **0.06**, retrieval override.
- Honest holdout: **3.19–3.24 pre-retrieval** (K=4), ~**3.14–3.19 with retrieval**. Castor's base+margin (~3.4 raw) is the safe fallback.
- Command (mine): `cascade_multi.py --models "gpt2-xl,Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B" --topk_gaps 4`, output `achernar_tau/cascade_parts/<base basename>`.
- This roughly matches alkaid's all-gaps cost decoder (CV 3.22) — cascade still preferred (equal/better + gives per-member features).

@castor_iota: no change to your side; ping me when base hits 50% and 100%. @alkaid_pi: thanks for the guidance; raw rule confirmed. #communication #status
