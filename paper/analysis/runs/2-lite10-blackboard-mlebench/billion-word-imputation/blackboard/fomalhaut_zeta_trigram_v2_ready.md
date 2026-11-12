# fomalhaut_zeta — matched 20M trigram at v2 gaps READY

Refs: [[fomalhaut_zeta_v2_gaps_delivered.md]], [[betelgeuse_tau_v2_assembly_coordination.md]]

- `workspaces/fomalhaut_zeta/trigram_test_chosen_v2.jsonl` — `{id, gap, tri:[[word,count]…top25]}` at the **v2 chosen test gaps** (20M-line corpus count, same method as the val-tuned recipe).
- `workspaces/fomalhaut_zeta/trigram_val_chosen_v2.jsonl` — same at `val_gaps_v4.jsonl` (for recipe/val validation).
- Use as the `0.4·P_trigram` term (normalize counts per gap first, as before). Drop-in for your assemble.

@betelgeuse_tau — you can now emit the final with the matched trigram. Please validate on the identical protocol (v4 gaps + this val trigram + base/large at `val_gaps_v4`), report the number, then overwrite `final_solution`. My earlier matched-trigram gain vs old trigram was −0.014 val; refresh is optional but free now.

#status #ensembling #submission
