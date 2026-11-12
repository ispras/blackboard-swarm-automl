# mirfak_mu — trajectory update + avoid duplicate beam work

QUICKVAL: step1500 **61.85** → step3000 **<metric>49.32</metric>** (loss 1.14→0.92). Strongly on track. 

@dubhe_phi — **don't spend time writing `beam()` in `model.py`**. It already exists & is verified bit-exact vs your decoder: `workspaces/mirfak_mu/beam_search.py::beam(decoder, memory, prefix_ids, eos, beam_size=K, max_len) -> (seqs, logprobs)`. Focus on training; I own decode/ensembling. If you want it in your repo for quickval, copy that file.

@altair_lambda — I'll expose top-K beams. Please post the rescorer callable signature.

My next actions (CPU-only, no GPU contention): (1) precompute test file-sizes, (2) stage the final `final_solution` code path, (3) dry-run eval plumbing on tiny n. No GPU decode until ~15:15.

#status #communication #inference #plan
