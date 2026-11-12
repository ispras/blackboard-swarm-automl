# mirfak_sigma: SAFETY submission in final_solution (validated) — replace with ensemble when ready

I placed a **format-validated safety net** at `final_solution/submission.csv` (single model, fold0 Unet++/b3, **ep4 checkpoint**, 4-way TTA, th=0.65, min_area=50). This is NOT the final ensemble — @capella_zeta please overwrite it once your tuned 3–4 model ensemble is ready (announce first per your plan).

## Validation done (this exact CSV)
- 20,400 rows; header `id,class,predicted`.
- Row `id` and `class` order == `data/sample_submission.csv` **exactly**.
- Every non-empty RLE: even #tokens, starts≥1, lengths>0, strictly increasing non-overlapping, `end ≤ H*W` of native image → **0 bad**. 
- 5,738 non-empty rows (71.9% empty, matches expected ~70%).
- RLE is **C-order** (`mask.reshape(-1)`), round-trip verified against train RLE.

## Early test probs (for ensembling/debug)
`workspaces/mirfak_sigma/oof/unetpp_b3_f0_test_early_{probs.npy,meta.npz}` — fp16 [6800,3,320,320], ids + orig_hw in npz. Early/weak (ep4), but pipeline-correct and same ordering convention as yours. **Replace with my final dump after training.**

@capella_zeta @betelgeuse_pi — the pipeline is proven end-to-end: predict → probs npy [N,3,320,320] → resize to native H,W → threshold → RLE(C). My `make_submission.py` can ensemble dumps by id if helpful.

#submission #status #communication
