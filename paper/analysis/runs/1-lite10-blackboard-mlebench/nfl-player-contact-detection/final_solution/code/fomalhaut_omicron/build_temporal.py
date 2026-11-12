"""feat2 -> feat3: add temporal lags/leads for key helmet & motion features."""
import pandas as pd, numpy as np, sys

BASE = 'workspaces/fomalhaut_omicron/cache/'
LAG_COLS = ['speed1', 'acceleration1', 'cy_e1', 'cy_s1', 'top_e1', 'height_e1',
            'hdist_e', 'hdist_s', 'closing', 'cy_e2', 'cy_s2']


def build(split):
    d = pd.read_parquet(BASE + 'feat2_%s.parquet' % split)
    d = d.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
    g = d.groupby(['game_play', 'p1', 'p2'], sort=False)
    new = {}
    for c in LAG_COLS:
        if c not in d.columns:
            continue
        s = d[c]
        for k in [1, 2, 3]:
            new['%s_m%d' % (c, k)] = s.groupby([d.game_play, d.p1, d.p2], sort=False).shift(k)
            new['%s_p%d' % (c, k)] = s.groupby([d.game_play, d.p1, d.p2], sort=False).shift(-k)
        new['%s_d1' % c] = s - new['%s_m1' % c]
        new['%s_dp1' % c] = new['%s_p1' % c] - s
        new['%s_rng5' % c] = (g[c].transform(lambda x: x.rolling(5, min_periods=1, center=True).max()) -
                              g[c].transform(lambda x: x.rolling(5, min_periods=1, center=True).min()))
    nd = pd.DataFrame(new, index=d.index)
    d = pd.concat([d, nd], axis=1)
    d.to_parquet(BASE + 'feat3_%s.parquet' % split, index=False)
    print(split, 'feat3 rows', len(d), 'cols', len(d.columns))


if __name__ == '__main__':
    build(sys.argv[1])
