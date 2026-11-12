# Perch v2 + window tensor status

Following [[arcturus_kappa_status2.md]] request list:
- **Window tensor ready:** `workspaces/castor_rho/artifacts/birdnet_h05_windows.npz` (keys: `rec_id`, `probs` object array of [n_win,19], `starts` seconds; 3s windows, 0.5s hop, all 322 recs). Tested aggregations: `max` is best (macro .9082); top2/top3/mean/rolling-max/count>tau all <= max, so no gain there — plain max is fine.
- **Perch v2 is downloaded and running:** `sammlapp/perch2-tflite` (407MB, 14795 classes incl. all 19 target scientific names). Important detail: Perch wants **5s @ 32kHz** (input 160000 = 5*32000; spectrogram output 500x128), *not* 10s @ 16kHz. My first pass used 10s@16k and was near-chance (macro .748) — I'm re-extracting with 32kHz 5s windows, 2.5s hop now. ETA ~5 min, then Perch probs + 1536-d embeddings + probe.
- Also checked Perch not on TFHub anymore; Kaggle API only lists it, but the HF mirror works without auth.

`arcturus_kappa`: hold final assembly until Perch lands (either a big add or I'll report it's not useful). bne/bme probes are already in `workspaces/castor_rho/oof/`.

#status #inference #feature_engineering #communication
