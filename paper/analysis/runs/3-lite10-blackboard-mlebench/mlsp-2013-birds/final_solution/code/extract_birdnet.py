import numpy as np, time, glob, tensorflow as tf
from huggingface_hub import hf_hub_download
ROOT = '/download_job_code/payload/sandbox/workspaces/pollux_epsilon'
p = hf_hub_download('sammlapp/BirdNET_v2.4', 'V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite', cache_dir=f'{ROOT}/hf_cache')
lp = hf_hub_download('sammlapp/BirdNET_v2.4', 'V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt', cache_dir=f'{ROOT}/hf_cache')
labels = [l.strip() for l in open(lp).read().splitlines() if l.strip()]
commons = [l.split('_', 1)[1] for l in labels]
want = ['Brown Creeper', 'Pacific Wren', 'Pacific-slope Flycatcher', 'Red-breasted Nuthatch', 'Dark-eyed Junco',
        'Olive-sided Flycatcher', 'Hermit Thrush', 'Chestnut-backed Chickadee', 'Varied Thrush', 'Hermit Warbler',
        "Swainson's Thrush", "Hammond's Flycatcher", 'Western Tanager', 'Black-headed Grosbeak',
        'Golden-crowned Kinglet', 'Warbling Vireo', "MacGillivray's Warbler", "Steller's Jay", 'Common Nighthawk']
idx = [commons.index(w) for w in want]
print('mapped', idx, flush=True)
wavs = np.load(f'{ROOT}/wavs_48k.npy')
it = tf.lite.Interpreter(model_path=p, num_threads=8); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]
starts = list(range(0, 8))
all_logits, sel_logits = [], []
t0 = time.time()
for ci, w in enumerate(wavs):
    win = []
    for s in starts:
        x = w[s*48000:(s+3)*48000][None, :].astype(np.float32)
        it.set_tensor(inp['index'], x); it.invoke()
        win.append(it.get_tensor(out['index'])[0])
    win = np.stack(win)
    all_logits.append(win.max(0))           # max over windows, full 6522
    sel_logits.append(win[:, idx])          # per-window logits for 19 species
    if ci % 50 == 0: print(ci, f'{time.time()-t0:.0f}s', flush=True)
np.savez_compressed(f'{ROOT}/birdnet.npz', logits19=np.stack(sel_logits), full_max=np.stack(all_logits), idx=np.array(idx))
print('done', np.stack(sel_logits).shape, flush=True)
