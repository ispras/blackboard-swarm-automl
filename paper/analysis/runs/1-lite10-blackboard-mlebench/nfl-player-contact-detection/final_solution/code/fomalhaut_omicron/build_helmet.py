"""Build step-level helmet geometry table from baseline helmets.
Output: cache/hel_{split}.parquet keyed by (game_play, step, nfl_player_id, view).
"""
import pandas as pd, numpy as np, os, sys
os.makedirs('workspaces/fomalhaut_omicron/cache', exist_ok=True)
FPS = 59.94
STEPF = FPS / 10.0


def play_offsets(trk_path, meta_path, plays):
    meta = pd.read_csv(meta_path)
    meta = meta[meta.view == 'Endzone'][['game_play', 'start_time']].drop_duplicates('game_play')
    meta['start_time'] = pd.to_datetime(meta.start_time)
    trk = pd.read_csv(trk_path, usecols=['game_play', 'nfl_player_id', 'step', 'datetime'])
    trk = trk[trk.game_play.isin(plays)]
    t0 = trk[trk.step == 0].drop_duplicates('game_play')[['game_play', 'datetime']]
    t0['datetime'] = pd.to_datetime(t0.datetime)
    m = meta.merge(t0, on='game_play', how='left')
    m['offset'] = (m.datetime - m.start_time).dt.total_seconds() * FPS
    off = dict(zip(m.game_play, m.offset))
    return off


def build(split):
    plays = None
    if split == 'train':
        labs = pd.read_csv('data/train_labels.csv', usecols=['game_play'])
        plays = set(labs.game_play.unique())
        hel_path = 'data/train_baseline_helmets.csv'
        trk_path = 'data/train_player_tracking.csv'
    else:
        ss = pd.read_csv('data/sample_submission.csv', usecols=['contact_id'])
        plays = set(ss.contact_id.str.rsplit('_', n=3).str[0].unique())
        hel_path = 'data/test_baseline_helmets.csv'
        trk_path = 'data/test_player_tracking.csv'
    meta_path = 'data/{}_video_metadata.csv'.format(split)
    off = play_offsets(trk_path, meta_path, plays)

    dfs = []
    for ch in pd.read_csv(hel_path, chunksize=2_000_000):
        ch = ch[ch.game_play.isin(plays)].copy()
        if len(ch) == 0:
            continue
        ch['offset'] = ch.game_play.map(off)
        ch['offset'] = ch.offset.fillna(297.7)
        ch['step'] = np.round((ch.frame - ch.offset) / STEPF).astype(int)
        ch['cx'] = ch.left + ch.width / 2.0
        ch['cy'] = ch.top + ch.height / 2.0
        ch['bottom'] = ch.top + ch.height
        ch['area'] = ch.width * ch.height
        dfs.append(ch[['game_play', 'step', 'view', 'nfl_player_id', 'cx', 'cy',
                       'top', 'bottom', 'width', 'height', 'area']])
    hel = pd.concat(dfs, ignore_index=True)
    g = hel.groupby(['game_play', 'step', 'view', 'nfl_player_id']).agg(
        cx=('cx', 'mean'), cy=('cy', 'mean'), top=('top', 'mean'),
        bottom=('bottom', 'mean'), width=('width', 'mean'), height=('height', 'mean'),
        area=('area', 'mean'), nframes=('cx', 'size')).reset_index()
    g.to_parquet('workspaces/fomalhaut_omicron/cache/hel_{}.parquet'.format(split), index=False)
    print(split, 'rows', len(g), 'plays', g.game_play.nunique())
    return g


if __name__ == '__main__':
    build(sys.argv[1])
