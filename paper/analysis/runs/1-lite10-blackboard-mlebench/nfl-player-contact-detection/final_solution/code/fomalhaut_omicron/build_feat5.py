"""feat4 + procyon relative helmet-pose (for p1,p2) + sirius ground-CNN prob -> feat5."""
import pandas as pd, numpy as np, sys

BASE = 'workspaces/fomalhaut_omicron/cache/'
RC = ['cy_rel_e', 'top_rel_e', 'h_rel_e', 'bottom_rel_e', 'cy_rank_e', 'h_rank_e',
      'cy_rel_s', 'top_rel_s', 'h_rel_s', 'bottom_rel_s', 'cy_rank_s', 'h_rank_s',
      'nbox_e', 'nbox_s']


def build(split):
    d = pd.read_parquet(BASE + 'feat4_%s.parquet' % split)
    rel = pd.read_parquet('workspaces/procyon_theta/cache/relpose_%s.parquet' % split)
    rel = rel[['game_play', 'step', 'player'] + RC].copy()
    for side in ['1', '2']:
        key = 'p' + side
        r = rel.rename(columns={c: c + '_' + side for c in RC})
        r = r.rename(columns={'player': key})
        r[key] = r[key].astype(str)
        d = d.merge(r, on=['game_play', 'step', key], how='left')
    for view in ['e', 's']:
        d['rel_cy_diff_' + view] = d['cy_rel_%s_1' % view] - d['cy_rel_%s_2' % view]
        d['rel_h_diff_' + view] = d['h_rel_%s_1' % view] - d['h_rel_%s_2' % view]
    cnn = pd.read_parquet('workspaces/sirius_tau/ground_cnn_feat_%s.parquet' % split)
    cnn = cnn.drop_duplicates('contact_id')
    d = d.merge(cnn, on='contact_id', how='left')
    d['ground_cnn_prob'] = d['ground_cnn_prob'].fillna(-10.0)
    d.to_parquet(BASE + 'feat5_%s.parquet' % split, index=False)
    print(split, 'feat5 rows', len(d), 'cols', len(d.columns))


if __name__ == '__main__':
    build(sys.argv[1])
