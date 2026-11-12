# arcturus_kappa — negative results (duplicates, XLNet, DeBERTa) + final v3 audit PASS

Re [[antares_upsilon_v3_written.md]] [[arcturus_kappa_v3b_cw_definition.md]] [[arcturus_kappa_v3_independent_repro.md]]

## Final submission audit (PASS)
`final_solution/submission.csv` (v3): pandas/csv parse OK — 302,644 rows, ids 0..302643 in order, no nulls, no control chars, header `id,"sentence"`, every changed row = exactly one token inserted at a valid gap, unchanged rows byte-identical to test, 265,447 inserts (87.71%). My independent holdout reproduction of the v3 recipe: **<metric>2.7619</metric>** (antares: 2.7609). v3 is safe.

## Negative results (so nobody re-spends time)
1. **Sentence-duplicate / retrieval is a dead end.** On a random 200k mid-corpus sample: *exact* duplicate instances = **5/200k (0.0025%)**. Prefix-5 reuse 3.8% / suffix-5 5.9% is only common-phrase overlap. The corpus is effectively deduplicated ⇒ nearest-neighbour/near-duplicate reconstruction of test originals from train will find ~nothing. Do not build retrieval.
2. **XLNet-large cloze is much worse** than roberta-large: at the true gap, expected-Lev word cost **3.09** (acc 0.36) vs roberta-large **1.92** (0.62); fusing it *hurts* (best rl⊕xl 2.17). Dropped.
3. **DeBERTa-v3-large has no released MLM head** (`cls.predictions.*` randomly initialised in HF checkpoint) ⇒ unusable for cloze. ELECTRA-large-generator works but is worse (2.40/0.50) and does not help the ensemble.
4. **4th detector (`large3`, 800k lines)**: best single (gap-acc 0.8597) but 4-way mix 0.8665 vs mix3 0.8656 → +0.0009, inside noise. Not used (as reported, v4 not worth it).
5. Trigrams clean add ~0; MLM gap-`maxp` adds ~0.0005–0.003; string candidates beyond prefix+suffix+del1+substr5 add ~0.001.

## Status
- I recommend **keeping v3** (validated 2.7609/2.7619). The v3b additive gate is ~0.004 per antares' reproduction — not worth the risk. `decisions_v3b_*` / `submission_v3b.csv` remain in my workspace if antares chooses to validate and adopt; otherwise ignore.
- I will not modify `final_solution`. I'm available until the end for independent verification of any further change antares/spica make.

#insight #ensembling #model_selection #postprocessing #status #communication #submission
