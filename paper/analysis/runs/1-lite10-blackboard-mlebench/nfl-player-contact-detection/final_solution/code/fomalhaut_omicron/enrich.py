"""Add per-player proximity/crowding + team/position/rolling features to feat_*.parquet -> feat2_*.parquet"""
import pandas as pd, numpy as np, sys, os

TOP_POS = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'DE', 'DT', 'NT', 'ILB', 'OLB', 'MLB',
           'CB', 'SS', 'FS', 'DB', 'LB', 'K', 'P', 'LS', 'FB', 'DL', 'OT', 'OG']


def load_track(split):
    p = 'data/{}_player_tracking.csv'.format(split)
    t = pd.read_csv(p, usecols=['game_play', 'step', 'nfl_player_id', 'x_position', 'y_position',
                                'team', 'position'])
    t = t.rename(columns={'x_position': 'x', 'y_position': 'y'})
    return t


def proximity(t):
    base = t[['game_play', 'step', 'nfl_player_id', 'x', 'y']]
    m = base.merge(base, on=['game_play', 'step'], suffixes=('_a', '_b'))
    m = m[m.nfl_player_id_a != m.nfl_player_id_b]
    dx = m.x_a - m.x_b; dy = m.y_a - m.y_b
    m['d'] = np.sqrt(dx * dx + dy * dy)
    g = m.groupby(['game_play', 'step', 'nfl_player_id_a']).agg(
        nn_dist=('d', 'min'), cnt2=('d', lambda s: (s < 2).sum()),
        cnt5=('d', lambda s: (s < 5).sum())).reset_index()
    g = g.rename(columns={'nfl_player_id_a': 'nfl_player_id'})
    return g


def build(split):
    base = 'workspaces/fomalhaut_omicron/cache/'
    d = pd.read_parquet(base + 'feat_{}.parquet'.format(split))
    t = load_track(split)
    prox = proximity(t)
    t2 = t[['game_play', 'step', 'nfl_player_id', 'team', 'position']]
    for side in ['1', '2']:
        key = 'p' + side
        t2r = t2.rename(columns={'nfl_player_id': key, 'team': 'team' + side,
                                 'position': 'pos' + side})
        t2r[key] = t2r[key].astype(str)
        d = d.merge(t2r, on=['game_play', 'step', key], how='left')
        pr = prox.rename(columns={'nfl_player_id': key, 'nn_dist': 'nn_dist' + side,
                                  'cnt2': 'cnt2' + side, 'cnt5': 'cnt5' + side})
        pr[key] = pr[key].astype(str)
        d = d.merge(pr, on=['game_play', 'step', key], how='left')
    d['same_team'] = (d.team1 == d.team2).astype(np.int8)
    d['same_team'] = d.same_team.where(d.p2 != 'G', -1)
    d = d.sort_values(['game_play', 'p1', 'p2', 'step'])
    grp = d.groupby(['game_play', 'p1', 'p2'], sort=False)
    for k in [5, 10, 20]:
        d['dmin_w%d' % k] = grp.dist.transform(lambda s: s.rolling(2 * k + 1, min_periods=1, center=True).min())
    # pos one-hot (coarse) for p1 and p2
    for side in ['1', '2']:
        col = 'pos' + side
        d[col + '_cat'] = pd.Categorical(d[col], categories=TOP_POS).codes.astype(np.int16)
    # drop raw string cols
    d = d.drop(columns=['team1', 'team2', 'pos1', 'pos2'])
    d.to_parquet(base + 'feat2_{}.parquet'.format(split), index=False)
    print(split, 'feat2 rows', len(d), 'cols', len(d.columns))


if __name__ == '__main__':
    build(sys.argv[1])
