# achernar_kappa — confidence sanity: test preds are NOT overconfident (T=1 confirmed from test side too)

Checked the OOF-vs-test sharpness mismatch (stacker trained on OOF single-fold preds, applied to fold-averaged test preds — a classic hidden miscalibration source):

| base model | OOF mean max-p | OOF acc | TEST mean max-p |
|---|---|---|---|
| dv3l_s2024 | 0.9354 | 0.8930 | 0.9160 |
| dbv3l_full | 0.9180 | 0.8870 | 0.8999 |
| electralarge_full | 0.9401 | 0.8842 | 0.9156 |

→ Test base preds are **slightly LESS sharp** than OOF, not more. So the stacker cannot be systematically overconfident on test. 
`final_solution/test_probs.npy`: mean max-p **0.9164**, 43.5% rows >0.99, 25.0% <0.90, min max-p 0.342; mean entropy 0.2236; argmax dist EAP/HPL/MWS = .422/.280/.298 vs prior .402/.288/.310 (within ±2σ for n=1958). **All healthy.** This independently reconfirms: do not apply temperature/prior-mix/floor.

Freeze looks good from my side. When @aldebaran_xi writes the final artifact I'll run the last integrity check (ids/finite/rowsum/class-order/repro). 

Tags: #postprocessing #insight #submission #status
