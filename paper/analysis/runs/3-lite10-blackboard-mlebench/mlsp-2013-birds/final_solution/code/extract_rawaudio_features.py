import os, numpy as np, pandas as pd, librosa, soundfile as sf

ROOT = '/download_job_code/payload/sandbox'
ED = f'{ROOT}/data/essential_data'

def load_ids():
    order = []
    for line in open(f'{ED}/rec_labels_test_hidden.txt'):
        line = line.strip()
        if line.startswith('rec_id'):
            continue
        order.append(int(line.split(',')[0]))
    return order

def load_map():
    m = {}
    for line in open(f'{ED}/rec_id2filename.txt'):
        line = line.strip()
        if line.startswith('rec_id'):
            continue
        rid, fn = line.split(',')
        m[int(rid)] = fn
    return m

def stats(a, axis=1, ps=(10, 25, 50, 75, 90)):
    out = [a.mean(axis), a.std(axis), a.max(axis), a.min(axis)]
    for p in ps:
        out.append(np.percentile(a, p, axis=axis))
    return np.concatenate([np.atleast_2d(o).reshape(-1) for o in out])

def extract(path):
    x, sr = sf.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    x = x - x.mean()
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak  # normalize amplitude (recorder gain varies)
    feats, names = [], []

    def add(v, nm):
        v = np.asarray(v).reshape(-1)
        feats.append(v)
        names.extend([f'{nm}_{i}' for i in range(len(v))])

    # A) log-mel stats
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=512,
                                       n_mels=96, fmin=50, fmax=8000, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    add(stats(logS), 'mel')
    # delta-mel stats (temporal change)
    dS = librosa.feature.delta(logS, order=1, width=9)
    add(np.concatenate([dS.mean(1), dS.std(1), np.abs(dS).max(1)]), 'dmeld')

    # B) MFCC
    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=40)
    add(np.concatenate([mfcc.mean(1), mfcc.std(1)]), 'mfcc')
    dm = librosa.feature.delta(mfcc, width=9)
    add(np.concatenate([dm.mean(1), dm.std(1)]), 'dmfcc')

    # C) frame-level spectral descriptors
    cen = librosa.feature.spectral_centroid(S=S, sr=sr)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    flat = librosa.feature.spectral_flatness(S=S)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=1024, hop_length=512)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=512)
    for nm, v in [('cen', cen), ('bw', bw), ('roll', roll), ('flat', flat), ('zcr', zcr), ('rms', rms)]:
        add(stats(v), nm)
    # D) spectral contrast
    con = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
    add(np.concatenate([con.mean(1), con.std(1)]), 'contrast')
    # E) tonnetz
    ton = librosa.feature.tonnetz(chroma=librosa.feature.chroma_stft(S=S, sr=sr))
    add(np.concatenate([ton.mean(1), ton.std(1)]), 'tonnetz')
    # F) temporal energy dynamics
    r = rms.reshape(-1)
    thr = np.percentile(r, 90) * 0.3
    act = (r > thr).astype(np.float32)
    # run lengths of active frames
    changes = np.diff(np.concatenate([[0], act, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    lens = (ends - starts).astype(np.float32)
    add([act.mean(), len(starts), lens.mean() if len(lens) else 0.0,
         lens.max() if len(lens) else 0.0, lens.std() if len(lens) else 0.0,
         np.median(r), np.percentile(r, 95)], 'dyn')
    return np.concatenate(feats).astype(np.float32), names

if __name__ == '__main__':
    order = load_ids()
    fnmap = load_map()
    X, names, ids = [], None, []
    for rid in order:
        f = f'{ED}/src_wavs/{fnmap[rid]}.wav'
        v, names = extract(f)
        X.append(v); ids.append(rid)
    X = np.stack(X)
    np.savez_compressed(f'{ROOT}/workspaces/pollux_beta/features_rawaudio.npz',
                        X=X, ids=np.array(ids), names=np.array(names))
    print('X', X.shape, 'nan', np.isnan(X).sum(), 'inf', np.isinf(X).sum())
