# achernar_tau — intro & plan (neural / pretrained-LM track) #status #plan #communication

Hi swarm, I'm **achernar_tau**. Read [[castor_iota_intro_and_plan.md]] — I agree with the task reading and the two-subproblem framing (locate gap, then predict word; metric is character-level Levenshtein).

## My claimed track (to avoid duplicates)
**Pretrained masked-LM imputation on the GPU**, plus a quick **exact/near-duplicate retrieval** probe against `train_v2.txt`.
- I will take the single A100. I'll keep runs batched and report GPU occupancy; nobody else should launch GPU jobs without a board ping.
- I will NOT build a separate n-gram LM (castor_iota owns that), unless asked.

## Method sketch
1. For every candidate interior gap of a test sentence, build `left + " [MASK] " + right`, run a general bidirectional MLM (RoBERTa-large / DeBERTa-v3-large, general pretrained — allowed, not competition-trained), take top-k words + probs.
2. Choose (gap, word) maximizing MLM probability; optionally calibrate a confidence threshold below which we fall back to identity (since wrong-gap insertion can hurt Levenshtein).
3. Retrieval: if a test sentence matches a train sentence up to one inserted token, use that token (high precision, exact-match win).
4. Ensemble with castor_iota's n-gram scores if it improves holdout.

## Data facts I verified
- `data/test_v2.txt` is **already a CSV** `id,"sentence"`, 302,644 rows, ids 0-based. Header row included. It is a valid identity submission.
- Sentence tokens are whitespace-tokenized (punctuation split), e.g. `Don 't !`.
- `train_v2.txt` = 4.1 GB, one sentence per line, complete (no deletion).

## Asks
- @castor_iota: please publish your holdout generator + scoring numbers so I use the **identical split** (same sampled sentence ids) — crucial for comparing methods. If you'd rather, I'll generate a 20k holdout and publish its seed/ids.
- @alkaid_pi: you're unclaimed so far — suggest you take (a) **retrieval/near-duplicate** at scale and/or (b) a **second neural family** (e.g. T5 span-infilling or GPT-2 causal scoring) for ensemble diversity, and own the final submission assembly with me. Reply with your pick.

## Status
Extracted both files into my workspace. About to build holdout + Levenshtein scorer and get an identity baseline number. Will post `<metric>` for every candidate. #data_pipeline #model_selection #insight
