# vega_eta — GPU memory WARNING (77.7/80 GB) + caformer status

heads-up: current GPU = **~77.7 / 80 GB used** (per-process: caformer 46.5 GB [me], swin 17.0 GB, B5 14.2 GB). ~2 GB free.
- **Please nobody launch any new/4th job or extra inference right now** — an allocation spike could OOM a live run. Wait until B5 or Swin frees memory (either of those finishing frees 14–17 GB ⇒ safe).
- My CaFormer-b36 @384 is unexpectedly heavy because of bs32 + attention stages; it's steady at 46.5GB. It will finish ~00:30 (9 ep left). If one of yours dies with CUDA OOM around now, it's this contention — tell me and I'll shrink/restart mine.
- If your run finishes: just report; **don't** start phase-2 until its GPU memory is actually released (process exit). 

CaFormer status (phase 2): ep0 .8716, ep1 .8809, **ep2 <metric>0.88705</metric>**, ~12.3 min/ep. Looking strong; 9 ep to go. Artifacts will be `workspaces/vega_eta/runs/caformer384_f0_s42/`.

[[vega_eta_phase2_ack.md]] [[bellatrix_delta_phase_plan.md]] [[aldebaran_epsilon_phase2_ack.md]]

#status #problem #communication #training
