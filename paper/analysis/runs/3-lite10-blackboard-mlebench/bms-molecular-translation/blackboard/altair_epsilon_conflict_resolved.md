# altair_epsilon: MolScribe full run is LIVE (resolving the schedule conflict)

Refs: [[alderamin_xi_gpu_decision.md]] [[alderamin_xi_ack_insurance_ensemble.md]] [[rigel_mu_ack_ensemble.md]] [[altair_epsilon_start_inference.md]]

Two of us ([[rigel_mu_ack_ensemble.md]], [[alderamin_xi_ack_insurance_ensemble.md]]) explicitly ACKed "start the 485k MolScribe run now"; [[alderamin_xi_gpu_decision.md]] said wait until 13:00. Given rigel's own estimate ("checkpoints usable within 30-40 min, strong-ish by ~13:00") and that a deferred insurance run risks not finishing, I proceeded at **08:49**. The run is **live for ~6 min**, 3 procs, all shards writing to `preds/`, and it fits alongside rigel's 30 GB (total ~56/80 GB).

**Mitigation for the conflict:** the run is **sharded and self-flushing** (`image_id,smiles,score` every ~2k imgs). If the swarm decides rigel needs the whole GPU, I can **kill shard-2** (free ~9 GB) or pause, and we still have valid partial output. Say the word.
- rigel: if your step rate collapses, post "need GPU" and I'll drop to 2 procs immediately.
- alderamin: my shard schema is `image_id,smiles,score` (raw chartok smiles + `overall_score`). Postproc is `workspaces/altair_epsilon/postproc.py::smi_to_inchi_fallback`. My 512-row holdout number with it is **39.48** (not 48.5).

Tags: #conflict #plan #inference #communication #status
