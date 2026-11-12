# alkaid_pi — evaluation protocol for the domain-LM dump (ready to run, no GPU) #model_selection #ensembling #plan

Agreed with [[achernar_tau_domain_lm_yes.md]]: achernar fine-tunes, I own the **honest, paired evaluation**. Harness is ready and already validated on a known case.

## What to dump (from achernar)
One jsonl per model, schema exactly like `rm_*_1000.jsonl`:
```
{"id": <holdout id>, "cands": [[gap, word, mean_logp, total_logp], ...]}
```
- Must cover the **same (gap,word) candidates** as `achernar_tau/cascade_holdout_k4_1997.jsonl` (top-4 gaps of base, MLM top-1 word). If you score all gaps, fine — I subset automatically.
- Ids: as many as feasible; the 4-member cascade holdout exists for ids **0–1990 (n=1,991)**. That is the max for a paired test. **Require n ≥ 1,000.**
- Also dump the **solo** scores; the harness reports solo + as-extra-member.

## Exact rule (same as final): raw equal blend + margin fallback
- equal **mean of member mean-log-probs**; pick max; if `best − second < tau` → identity; `tau` swept `0..0.30` on the same ids.
- Also report gap top-1 and paired bootstrap 95% CI of `(4+extra − 4)` on per-sentence Levenshtein.

## Decision criterion (pre-registered, to avoid small-sample noise like Qwen3B)
**ADD only if**: n ≥ 1,000 **and** paired `(4+extra − 4)` mean < 0 **and** bootstrap 95% CI upper bound < 0. Otherwise reject.
Baseline for reference on the 998-id overlap of `rm_gptj_1000`: 4-member blend 3.1012 vs +gpt-j 3.1423 ⇒ diff **+0.041, CI [−0.062, +0.144] → reject**. That is the noise floor you must beat.

CLI (I run it):
```bash
pyenv shell sis-black && python workspaces/alkaid_pi/eval_extra_member.py \
  --casc workspaces/achernar_tau/cascade_holdout_k4_1997.jsonl \
  --extra workspaces/achernar_tau/<domain_lm_scores>.jsonl --boot 2000
```
If solo is dramatically better (e.g. gap top-1 > 0.75), we also test it as the **base** of a rerank, not just an add-on.

## Integration (if it passes)
- Score the test top-4 cascade candidates with the fine-tuned LM (`cascade_multi.py --models <local_path>` works, or a tiny scorer); write `cascade_parts_plus/<basename>`.
- Re-assemble with the extra member in the blend. **Only after** a valid 4-member cascade submission is already banked in `final_solution/`.
- Deadline guard: if not passing by **02:30**, abandon and ship the validated 4-member cascade.

No action from @castor_iota needed until/unless it passes; I'll post the paired numbers and a recommendation.
