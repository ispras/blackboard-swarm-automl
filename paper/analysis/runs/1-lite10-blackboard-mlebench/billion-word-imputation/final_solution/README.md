# Billion Word Imputation — solution

Task: each test sentence has exactly one interior whitespace token deleted; reconstruct the
sentence. Metric: mean **character-level Levenshtein distance** (lower is better).

`final_solution/submission.csv` has 302,644 rows (`id,"sentence"`, ids 0..302643, CSV quotes
doubled) and is produced by the pipeline below from `data/train_v2.txt` + `data/test_v2.txt` only.

## Final pipeline (6-way blend: 1 base + 5 rerankers, 2 of them domain-adapted)

1. **Exact retrieval override** (distance 0, safe): index train sentences by prefix/suffix token
   anchors (first/last 12/8/5 tokens); if a train sentence equals the masked test sentence plus
   exactly one token, restore that token and gap with certainty. Recovers **2,928 / 302,644**
   test sentences (0.967%). Code: `code/retrieve_full.py`, `code/make_submission_retrieval.py`.
2. **Masked-LM candidate word per gap** (`roberta-large`, general pretrained): for every interior
   gap of every test sentence insert `[MASK]` and keep the top-8 words + probabilities
   (`code/mlm_test.py`; `code/mlm_short.py` handles the 533 two-token sentences).
3. **Base reranker over all gaps** (`gpt2-large`, general pretrained): for every gap, insert the
   MLM top-1 word and score the reconstructed sentence by mean token log-probability
   (`code/rerank_test.py`). Cheap enough to score all ~22 gaps of all 302k sentences.
4. **Cascade rerank of the top-4 gaps** (`gpt2-xl`, `Qwen2.5-1.5B`, `SmolLM2-1.7B`, plus two
   **in-domain fine-tuned** LMs): rescore the same reconstructed candidates with five complementary
   causal LMs (`code/cascade_multi.py`, `code/add_member.py`):
   - `gpt2-xl`, `Qwen/Qwen2.5-1.5B`, `HuggingFaceTB/SmolLM2-1.7B` (general pretrained),
   - **`ft_gpt2medium`**: `gpt2-medium` fine-tuned 1 epoch on 200 MB of `train_v2.txt`,
   - **`ft_gpt2large`**: `gpt2-large` fine-tuned 1 epoch on 150 MB of `train_v2.txt`
     (`code/finetune_lm.py` — training only on the provided train set, no test labels).
   Domain adaptation on this news corpus was the single biggest modelling win: the fine-tuned
   `gpt2-medium` scores **3.14** solo on holdout (vs ~3.37 for off-the-shelf `gpt2-xl`).
   The final score per candidate is the **equal mean of the 6 mean-log-probabilities**
   (base + 5 rerankers).
5. **Decision rule** (`code/assemble_submission.py`): take the highest-scoring candidate; if its
   score margin over the second-best gap is `< 0.06`, fall back to the identity submission
   (masked sentence unchanged). Wrong-gap insertion is much worse than identity, so the fallback
   is essential. Retrieval overrides are applied first; missing ids fall back to identity.

## Validation (holdout carved from `train_v2.txt`; no test labels used)
- identity baseline: **5.52** mean char Levenshtein (n=9,972)
- oracle gap + fixed short word: ~4.15
- single `gpt2-xl`/`gpt2-large` rerank + margin fallback: ~3.37–3.40
- 4-member cascade (base+xl+qwen+smol) + margin: **3.186** (n=1,991, pre-retrieval)
- **5-member cascade + domain `ft_gpt2medium`: 3.125** (paired paired Δ −0.062, 95% CI excludes 0)
- **final 6-member + domain `ft_gpt2large`: 3.088** (leakage-aware unseen-ids subset **3.168**;
  paired Δ vs 5-member −0.037 all / −0.026 unseen, CI includes 0 at n≈1,700 but point estimate
  negative at every margin threshold on both seen+unseen; added as a variance-reducing extra member)
- with retrieval override (0.967% of test ids): expected test ≈ **3.02–3.08**

## Reproduce
```
# 0) unzip data; pyenv shell sis-black
python code/mlm_test.py                 # roberta-large, all interior gaps, top-8 (+ mlm_short.py)
python code/rerank_test.py --model gpt2-large --outdir rerank_base_parts
python code/cascade_multi.py --base_glob 'rerank_base_parts/part_*.jsonl' --out_dir cascade_parts \
       --topk_gaps 4 --models 'gpt2-xl,Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B'
python code/finetune_lm.py --model gpt2-medium --train train_v2.txt --out ft_gpt2medium \
       --max_bytes 200000000 --epochs 1
python code/finetune_lm.py --model gpt2-large  --train train_v2.txt --out ft_gpt2large  \
       --max_bytes 150000000 --epochs 1
python code/add_member.py --cand_glob 'cascade_parts/part_*.jsonl' --out_dir cascade_parts_5m --model ft_gpt2medium
python code/add_member.py --cand_glob 'cascade_parts_5m/part_*.jsonl' --out_dir cascade_parts_6m --model ft_gpt2large
python code/retrieve_full.py ; python code/make_submission_retrieval.py
python code/assemble_submission.py --rerank rerank_base_parts/part_*.jsonl cascade_parts_6m/part_*.jsonl \
       --margin 0.06 --retrieval retrieval_recovered.tsv --out submission.csv
```
(On the orchestration machine these ran in per-agent workspaces; the scripts here are the exact
code used, with the same file formats. `code/verify_submission.py` checks coverage/quoting.)

## External pretrained models (all general-purpose; none trained on this competition's labels)
| model | source | revision | license | purpose |
|---|---|---|---|---|
| `roberta-large` | Hugging Face `roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | MIT | masked-LM candidate words per gap |
| `gpt2-large` | Hugging Face `gpt2-large` | `32b71b12589c2f8d625668d2335a01cac3249519` | MIT | base all-gap reranker / parent of `ft_gpt2large` |
| `gpt2-medium` | Hugging Face `gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | MIT | parent of `ft_gpt2medium` |
| `gpt2-xl` | Hugging Face `gpt2-xl` | `15ea56dee5df4983c59b2538573817e1667135e2` | MIT | cascade reranker |
| `Qwen2.5-1.5B` | Hugging Face `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0 | cascade reranker |
| `SmolLM2-1.7B` | Hugging Face `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | Apache-2.0 | cascade reranker |
| `ft_gpt2medium` | fine-tuned locally from `gpt2-medium` on `data/train_v2.txt` (200 MB, 1 epoch) | — | MIT-derived | in-domain cascade reranker |
| `ft_gpt2large` | fine-tuned locally from `gpt2-large` on `data/train_v2.txt` (150 MB, 1 epoch) | — | MIT-derived | in-domain cascade reranker |

Additional exploratory models (tested, not used in the final submission): `EleutherAI/gpt-neo-2.7B`,
`EleutherAI/gpt-j-6b`, `EleutherAI/pythia-2.8b`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-3B`
(general pretrained, same allowed categories). `rapidfuzz` is used for the Levenshtein metric.

No external data was used for training; everything is built only from `data/train_v2.txt`.
No test labels were read or used.
