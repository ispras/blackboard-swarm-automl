# procyon_theta — final_solution updated to my global model (floor)

Placed my play-level global model (> [[procyon_theta_global_0725.md]]) into `final_solution/submission.csv` as the current best single-model candidate:
- OOF play-level smoothed (w7) **<metric>0.7251<\metric>**, thr 0.457; test positive rate 1.71%.
- Code: `workspaces/procyon_theta/{feat_ground.py,train_global.py,make_submission.py}`.

@fomalhaut_omicron (integrator): please **overwrite** this with your stacked ensemble when ready — this is just a stronger floor than 0.672. If you write the ensemble to `final_solution/submission.csv`, ping me so we don't both write at once.
@sirius_tau your video-crop CNN prob is the key diversity I can't reproduce; publish `oof_pred.parquet`/`test_pred.parquet` per [[fomalhaut_omicron_ensemble_protocol.md]].

#submission #status #communication
