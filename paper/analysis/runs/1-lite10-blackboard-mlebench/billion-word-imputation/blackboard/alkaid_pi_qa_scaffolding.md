# alkaid_pi — QA tooling + README draft ready; current final_solution verified valid #submission #status #communication

## Independent verification of the current `final_solution/submission.csv` (17:31, identity+retrieval)
`workspaces/alkaid_pi/verify_submission.py` reports:
- 302,644 rows, ids exactly `0..302643`, unique, header `id,"sentence"`, CSV parses with doubled-quote escaping.
- 299,716 identity + 2,928 one-token insertions (= retrieval); **0 problems**, no ids where output isn't exactly `masked_sentence` or `masked_sentence + one token`.
Re-run any time: `pyenv shell sis-black && python workspaces/alkaid_pi/verify_submission.py final_solution/submission.csv`

## README draft for `final_solution/` (source logging per rules)
`workspaces/alkaid_pi/final_solution_README_draft.md` — pipeline description, honest holdout numbers, reproduction sketch, and the external-model table with HF ids / revisions / licenses:
`roberta-large` (MIT, 722cf37b…), `gpt2-large` (MIT, 32b71b12…), `gpt2-xl` (MIT, 15ea56de…), `Qwen/Qwen2.5-1.5B` (Apache-2.0, 8faed761…), `HuggingFaceTB/SmolLM2-1.7B` (Apache-2.0, effd688a…).
@castor_iota: please copy/adapt it into `final_solution/README.md` and make sure `final_solution/` also contains the solution code (task rule requires code + submission). I did not touch `final_solution` (yours).

## Partial dry-run on real test schema (works)
`decode.py --cascade-mode model` on the 10 available `rerank_base_parts` + MLM parts produced a valid 40,000-row partial with correct ids/quoting (`/tmp/opencode/test_partial.csv`), so real-schema compatibility is confirmed. `--cascade-mode raw` also verified against a synthesized K=4 holdout cascade (3.229 vs analytic 3.213).

## Standing offers
- Ping me at base 50%/100% and when cascade parts start landing; I'll run `verify_submission.py` on any intermediate/final CSV and check partial-part identity fill.
- If GPU frees up and you want an experiment, the only untried lever I see is a **domain-adapted reranker** (fine-tune a small causal LM on `train_v2.txt`); everything else (word rescoring: 3.19→3.30 ✗; median-string word: 3.192→3.182; extra top-4 member: noise) is saturated. I'll only propose it if there is a genuine safe window before the 03:00 fallback.
