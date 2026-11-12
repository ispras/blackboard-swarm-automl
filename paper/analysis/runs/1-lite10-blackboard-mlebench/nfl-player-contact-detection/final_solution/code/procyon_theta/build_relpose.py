import os, sys, time
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
HEL=os.path.join(HERE,'..','fomalhaut_omicron','cache')

def rel_features(hel):
    # per (game_play, step, view) medians/stds over all player boxes
    g=hel.groupby(['game_play','step','view'])
    agg=g.agg(med_cy=('cy','median'),std_cy=('cy','std'),med_top=('top','median'),
              med_h=('height','median'),std_h=('height','std'),
              med_bottom=('bottom','median'),nbox=('cy','size')).reset_index()
    h=hel.merge(agg,on=['game_play','step','view'],how='left')
    h['cy_rel']=(h.cy-h.med_cy)/(h.std_cy.fillna(0)+1e-3)
    h['top_rel']=(h.top-h.med_top)/(h.std_cy.fillna(0)+1e-3)
    h['h_rel']=(h.height-h.med_h)/(h.std_h.fillna(0)+1e-3)
    h['bottom_rel']=(h.bottom-h.med_bottom)/(h.std_cy.fillna(0)+1e-3)
    h['cy_rank']=h.groupby(['game_play','step','view'])['cy'].rank(pct=True)
    h['h_rank']=h.groupby(['game_play','step','view'])['height'].rank(pct=True)
    return h

def main(split):
    hel=pd.read_parquet(os.path.join(HEL,f'hel_{split}.parquet'))
    h=rel_features(hel)
    # restrict to Endzone/Sideline
    cols=['cy_rel','top_rel','h_rel','bottom_rel','cy_rank','h_rank','cy','height','top','nbox']
    out=None
    for view,tag in [('Endzone','e'),('Sideline','s')]:
        hv=h[h.view==view][['game_play','step','nfl_player_id']+cols].copy()
        hv=hv.rename(columns={c:f'{c}_{tag}' for c in cols})
        out=hv if out is None else out.merge(hv,on=['game_play','step','nfl_player_id'],how='outer')
    out=out.rename(columns={'nfl_player_id':'player'})
    out.to_parquet(os.path.join(CACHE,f'relpose_{split}.parquet'))
    print(split,'relpose',out.shape)

if __name__=='__main__':
    main('train'); main('test')
