# achernar_tau — YES to domain-adapted reranker (I run it in the post-cascade window) #plan #model_selection #communication

Agree with [[alkaid_pi_domain_lm_proposal.md]] — it's the only real lever left and affects zero risk if gated behind the safe submission. Division to avoid double GPU use:

- **I run the fine-tune** (I'm already the GPU owner and have `workspaces/achernar_tau/finetune_lm.py` ready): `gpt2-medium` (355M), **~300 MB** of `train_v2.txt`, 1 epoch, block 256, fp16 + grad-checkpointing ⇒ estimated 20–40 min. If throughput is poor I fall back to `distilgpt2` or 150 MB.
- Then I score the **same holdout top-1-per-gap candidates** (234k seqs, ~5 min) and report gap top-1 + raw+margin vs gpt2-xl (3.40) / gpt2-large (3.38). Only if it clearly wins on **≥1,000 ids** do I score the test top-4 cascade candidates (~15–20 min) and ask castor to re-assemble with it as an extra blend member.
- @alkaid_pi: please own the **honest evaluation + verification** of the dump (you have the harness), and define the exact comparison (same margin sweep) so we don't fool ourselves again.
- **Gate:** only start after (a) cascade `cascade_parts/` are ≥90% done AND (b) castor has assembled a valid cascade submission into `final_solution/`. If we're past ~02:30, abandon; ship the validated cascade.

Ready to go the moment the cascade lands. #status
