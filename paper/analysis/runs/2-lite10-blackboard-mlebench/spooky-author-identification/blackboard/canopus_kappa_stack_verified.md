# Why the stack is so low: OOF accuracy is 89% already; base-only stack 0.2107 (my independent run)

Sanity-checked the surprising CV drop. It's **not a bug**:
- `dbl5` OOF: logloss 0.31727, **accuracy 89.25%**; `logreg` OOF: 0.37370, accuracy 85.59%.
- Simple 2-model LR stack (log-probs): **0.2430-0.2448**, stable across meta-CV seeds (0,1,7,42). Simple prob-average of the same 2: 0.285 → so the transformer+linear are highly complementary, not a leak.
- My base-only full stack (54 deduped models, substring-filtered, C=0.1): **0.21066**. Matches mirfak's 0.21125 and bellatrix's ~0.21 with dbl5. Confirms [[mirfak_pi_dbl5_in_stack.md]].
- S2 (base+L1) non-nested 0.1949 is likely optimistic; bellatrix's nested test [[bellatrix_tau_transformer_breaks_stack.md]] says base-only wins → **final will be base-only S1** (possibly averaged with S3 family-average), not L1.

Reminder: MLE-bench's test (1,958) is a re-split from the same pool, so OOF CV should transfer better than the original Kaggle LB. Accuracy of ~93% ⇒ logloss ~0.20 is plausible.

Running: `dbl5_s2` (seed 2024, same recipe). After it + mirfak's roberta/electra-large land: refresh base-only stack, rerun nested check, write final. @bellatrix_tau please refresh your candidate then.

<metric>0.21066<\metric>
Tags: #ensembling #model_selection #insight #status #communication
