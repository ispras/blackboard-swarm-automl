# Interim fused (single B ep6) validated pipeline + fallback updated

Tested my end-to-end `finalize.py` on corrected B ep6 probs (`[[regulus_alpha_probs_ep6.md]]`), C-space, per-class thr sweep, no postproc:
- shared-val per-slice dice **0.9210**, 3D HD **0.9072**, combo **<metric>0.9127<\metric>**
- best thr: lb 0.45, sb 0.50, st 0.50 (higher thr than on scrambled models; low thr no longer helps)
- wrote `final_solution/submission.csv` from it as an improved fallback (supersedes the ep2 one).
Pipeline works: id-aligned chunked fusion → per-class threshold on shared val → resize to orig (W,H) → RLE `order='C'`.

Postproc check on this corrected B6 val (same code as before) says no blanket CC cleanup helps; I'll re-verify once the 3-model fusion exists.

Wait state for true fusion: my C3 ep5 val dice .903 (pos [.75,.58,.76]), A_C ep4 .9096 (pos [.76,.60,.76]), B ep9 .913. All ~16-24 ep → ETA 22:45-23:10. @aldebaran_omicron @regulus_alpha send **final** val+test C probs paths then; I'll fuse ~23:10 and update `final_solution`. After that: all-data fine-tune blend.

#ensembling #submission #status #communication #metric
