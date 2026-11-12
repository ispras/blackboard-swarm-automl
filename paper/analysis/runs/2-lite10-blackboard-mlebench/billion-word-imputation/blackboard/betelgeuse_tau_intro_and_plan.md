# betelgeuse_tau — intro + key metric insight + proposed division

Hi team, I'm betelgeuse_tau. Task: **Billion Word Imputation**. For each test sentence one word (never first, last is always `.`) was deleted; we must re-insert it. Metric = **mean Levenshtein distance** to the original sentence (lower is better). 302,645 test sentences; train ~4.1 GB text (~30M sentences).

## Key insight about the metric (please discuss)
The submission is free text, **not constrained to insert a real word**. The deleted word is inserted into `T = prefix + " " + suffix`. If we insert string `x` at the right gap, then
`Levenshtein(original, ours) ≈ Levenshtein(w_true, x)`.
So:
- submitting `test.txt` unchanged costs `mean(len(w_true))` (~4–5), i.e. the trivial baseline.
- inserting a short string costs `≈ len(w) + len(x) - 2*matched`; **when unsure, a short string is much safer than a full word** (a wrong long word adds its full length).
- Therefore the optimal submission is a **decision-theory problem**: choose gap `g` and string `x` minimizing expected char edit distance over the LM's candidate distribution — not necessarily the argmax word.

## Proposed plan / roles (reply with preference)
1. **betelgeuse_tau (me)**: validation harness (holdout from train, task-identical deletion), fast Levenshtein scorer, decision-theoretic insertion optimizer, keep shared baseline in `final_solution`.
2. **fomalhaut_zeta**: n-gram language model (counts/backoff, custom or kenlm-style) → per-gap candidate word distributions for location+word.
3. **mirfak_chi**: pretrained masked LM (e.g. RoBERTa/BERT MLM from HF) → candidate words per gap, and/or a location model.
We must keep models **diverse** and compare on the *same* holdout. I'll publish the holdout + scorer format next.

## Baseline to beat
I'll publish exact numbers for (a) unchanged sentence, (b) single short char at random gap, (c) oracle-gap + short string, on a 20k holdout, within ~15 min. Please post your #status and which role you take.

#plan #communication #insight #submission
