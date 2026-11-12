"""Build pair-level feature table = tracking kinematics + helmet geometry."""
import pandas as pd, numpy as np, os, sys, gc
os.makedirs('workspaces/fomalhaut_omicron/cache', exist_ok=True)


def load_track(split):
    if split == 'train':
        t = pd.read_csv('data/train_player_tracking.csv')
    else:
        t = pd.read_csv('data/test_player_tracking.csv')
    t = t.rename(columns={'x_position': 'x', 'y_position': 'y'})
    t['vx'] = t.speed * np.cos(np.deg2rad(t.direction))
    t['vy'] = t.speed * np.sin(np.deg2rad(t.direction))
    t['ox'] = np.cos(np.deg2rad(t.orientation))
    t['oy'] = np.sin(np.deg2rad(t.orientation))
    return t[['game_play', 'step', 'nfl_player_id', 'x', 'y', 'speed', 'distance',
              'direction', 'orientation', 'acceleration', 'sa', 'vx', 'vy', 'ox', 'oy']]


TC = ['x', 'y', 'speed', 'distance', 'acceleration', 'sa']


def pair_feats(lab, trk, hel):
    g = lab.copy()
    g['p1'] = g.nfl_player_id_1.astype(str)
    g['p2'] = g.nfl_player_id_2.astype(str)
    t1 = trk.rename(columns={c: c + '1' for c in trk.columns if c not in ('game_play', 'step', 'nfl_player_id')})
    t1 = t1.rename(columns={'nfl_player_id': 'p1'})
    t1['p1'] = t1.p1.astype(str)
    g = g.merge(t1, on=['game_play', 'step', 'p1'], how='left')
    t2 = trk.rename(columns={c: c + '2' for c in trk.columns if c not in ('game_play', 'step', 'nfl_player_id')})
    t2 = t2.rename(columns={'nfl_player_id': 'p2'})
    t2['p2'] = t2.p2.astype(str)
    g = g.merge(t2, on=['game_play', 'step', 'p2'], how='left')

    dx = g.x1 - g.x2
    dy = g.y1 - g.y2
    g['dist'] = np.sqrt(dx * dx + dy * dy)
    g['dx'] = dx
    g['dy'] = dy
    rvx = g.vx1 - g.vx2
    rvy = g.vy1 - g.vy2
    g['rel_speed'] = np.sqrt(rvx * rvx + rvy * rvy)
    g['closing'] = -(dx * rvx + dy * rvy) / (g.dist + 1e-6)
    g['speed_sum'] = g.speed1 + g.speed2
    g['speed_diff'] = (g.speed1 - g.speed2).abs()
    g['acc_sum'] = g.acceleration1 + g.acceleration2
    cosA = (dx * g.vx1 + dy * g.vy1) / (g.dist * g.speed1 + 1e-6)
    g['p1_toward'] = cosA
    cosB = (-dx * g.vx2 - dy * g.vy2) / (g.dist * g.speed2 + 1e-6)
    g['p2_toward'] = cosB

    # lags / leads of distance
    g = g.sort_values(['game_play', 'p1', 'p2', 'step'])
    grp = g.groupby(['game_play', 'p1', 'p2'], sort=False)
    for k in [1, 2, 3]:
        g['dist_m%d' % k] = grp.dist.shift(k)
        g['dist_p%d' % k] = grp.dist.shift(-k)
    g['dist_d1'] = g.dist - g.dist_m1
    g['dist_dp1'] = g.dist_p1 - g.dist
    g['dist_min_w5'] = g.groupby(['game_play', 'p1', 'p2'], sort=False).dist.transform(
        lambda s: s.rolling(5, min_periods=1, center=True).min())
    g = g.drop(columns=['vx1', 'vy1', 'ox1', 'oy1', 'vx2', 'vy2', 'ox2', 'oy2'])

    # ---- helmet merges ----
    for view, suf in [('Endzone', '_e'), ('Sideline', '_s')]:
        h = hel[hel.view == view].drop(columns=['view'])
        h1 = h.rename(columns={c: c + suf + '1' for c in h.columns if c not in ('game_play', 'step', 'nfl_player_id')})
        h1 = h1.rename(columns={'nfl_player_id': 'p1'})
        h1['p1'] = h1.p1.astype(str)
        g = g.merge(h1, on=['game_play', 'step', 'p1'], how='left')
        h2 = h.rename(columns={c: c + suf + '2' for c in h.columns if c not in ('game_play', 'step', 'nfl_player_id')})
        h2 = h2.rename(columns={'nfl_player_id': 'p2'})
        h2['p2'] = h2.p2.astype(str)
        g = g.merge(h2, on=['game_play', 'step', 'p2'], how='left')
        g['hdist' + suf] = np.sqrt((g['cx' + suf + '1'] - g['cx' + suf + '2']) ** 2 +
                                   (g['cy' + suf + '1'] - g['cy' + suf + '2']) ** 2)
        # box overlap from center/width/height
        l1 = g['cx' + suf + '1'] - g['width' + suf + '1'] / 2
        r1 = g['cx' + suf + '1'] + g['width' + suf + '1'] / 2
        u1 = g['cy' + suf + '1'] - g['height' + suf + '1'] / 2
        d1 = g['cy' + suf + '1'] + g['height' + suf + '1'] / 2
        l2 = g['cx' + suf + '2'] - g['width' + suf + '2'] / 2
        r2 = g['cx' + suf + '2'] + g['width' + suf + '2'] / 2
        u2 = g['cy' + suf + '2'] - g['height' + suf + '2'] / 2
        d2 = g['cy' + suf + '2'] + g['height' + suf + '2'] / 2
        iw = (np.minimum(r1, r2) - np.maximum(l1, l2)).clip(lower=0)
        ih = (np.minimum(d1, d2) - np.maximum(u1, u2)).clip(lower=0)
        inter = iw * ih
        union = g['area' + suf + '1'] + g['area' + suf + '2'] - inter
        g['hiou' + suf] = inter / (union + 1e-6)
        g['hoverlap' + suf] = inter / (np.minimum(g['area' + suf + '1'], g['area' + suf + '2']) + 1e-6)
    return g


def build(split):
    if split == 'train':
        lab = pd.read_csv('data/train_labels.csv')
    else:
        lab = pd.read_csv('data/sample_submission.csv')
        lab['game_play'] = lab.contact_id.str.rsplit('_', n=3).str[0]
        lab['step'] = lab.contact_id.str.rsplit('_', n=3).str[1].astype(int)
        lab['nfl_player_id_1'] = lab.contact_id.str.rsplit('_', n=3).str[2].astype(int)
        lab['nfl_player_id_2'] = lab.contact_id.str.rsplit('_', n=3).str[3]
    trk = load_track(split)
    hel = pd.read_parquet('workspaces/fomalhaut_omicron/cache/hel_{}.parquet'.format(split))
    g = pair_feats(lab, trk, hel)
    g.to_parquet('workspaces/fomalhaut_omicron/cache/feat_{}.parquet'.format(split), index=False)
    print(split, 'feat rows', len(g), 'cols', len(g.columns))
    return g


if __name__ == '__main__':
    build(sys.argv[1])
