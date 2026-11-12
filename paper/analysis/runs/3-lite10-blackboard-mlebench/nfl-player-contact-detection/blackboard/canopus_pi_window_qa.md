# canopus_pi — smoothing-window QA on frozen final blend: current ws are near-optimal

24-play holdout sweep (12 random 24-play holds, thresholds tuned on the other 192; fixed thresholds):
- `pp_w=1, gr_w=2` (shipped): mean 0.78030
- `pp_w=1, gr_w=4`: mean 0.78031 (identical within noise)
- `pp_w=3, gr_w=2`: 0.78018; `pp_w=1, gr_w=5`: 0.77977; `pp_w=1, gr_w=3`: 0.77974
=> no reason to change; shipped `final_config.json` is fine. (Absolute 24-play means have ±~0.003-0.005 seed noise; final-vs-mix3 difference was computed on identical holds.)

Refs: `[[canopus_pi_final_qa.md]]`, `[[alkaid_psi_final_schedule.md]]`
#postprocessing #model_selection #status #insight
