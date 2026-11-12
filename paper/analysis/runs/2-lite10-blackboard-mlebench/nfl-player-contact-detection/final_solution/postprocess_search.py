import numpy as np, pandas as pd, json, numba

KEYS = ['game_play','nfl_player_id_1','nfl_player_id_2','step']
GKEYS = ['game_play','nfl_player_id_1','nfl_player_id_2']


@numba.njit(cache=True)
def _bounds(gid):
    n = len(gid)
    gs = np.zeros(n, dtype=np.int64); ge = np.zeros(n, dtype=np.int64)
    s = 0
    for i in range(1, n):
        if gid[i] != gid[i-1]:
            for j in range(s, i): ge[j] = i
            s = i
    for j in range(s, n): ge[j] = n
    cur = 0
    for i in range(1, n):
        if gid[i] != gid[i-1]:
            cur = i
        gs[i] = cur
    return gs, ge


@numba.njit(cache=True)
def _roll_mean_center(vals, gs, ge, w):
    n = len(vals); out = np.empty(n, dtype=np.float32)
    h = w // 2
    for i in range(n):
        lo = i - h
        if lo < gs[i]: lo = gs[i]
        hi = i + h + 1
        if hi > ge[i]: hi = ge[i]
        s = 0.0
        for j in range(lo, hi): s += vals[j]
        out[i] = s / (hi - lo)
    return out


def smooth(df, prob, w):
    if w <= 1:
        return df, prob.astype(np.float32)
    d = df.copy()
    d['_p'] = prob.astype(np.float32)
    d = d.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    gid = d.groupby(GKEYS, sort=False).ngroup().values.astype(np.int64)
    gs, ge = _bounds(gid)
    sm = _roll_mean_center(d['_p'].values, gs, ge, w)
    d['_p'] = sm
    return d, sm


def counts_sorted(ps, ys, thr):
    k = int(np.searchsorted(-ps, -thr, side='right'))
    tp = int(ys[:k].sum()); fp = k - tp
    pos = int(ys.sum()); neg = len(ys) - pos
    return tp, fp, pos - tp, neg - fp


def joint_mcc(pp, yp, pg, yg, nq=160):
    gp_t = np.unique(np.quantile(pp, np.linspace(0.85, 0.9995, nq)))
    gg_t = np.unique(np.quantile(pg, np.linspace(0.85, 0.9995, nq)))
    op_ = np.argsort(-pp); pp_s = pp[op_]; yp_s = yp[op_].astype(np.int64)
    og_ = np.argsort(-pg); pg_s = pg[og_]; yg_s = yg[og_].astype(np.int64)
    cp = [counts_sorted(pp_s, yp_s, t) for t in gp_t]
    cg = [counts_sorted(pg_s, yg_s, t) for t in gg_t]
    best = (-1, 0, 0)
    for i, a in enumerate(cp):
        for j, b in enumerate(cg):
            tp = float(a[0]+b[0]); fp = float(a[1]+b[1]); fn = float(a[2]+b[2]); tn = float(a[3]+b[3])
            den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            if den == 0: continue
            m = (tp*tn - fp*fn)/den
            if m > best[0]: best = (m, float(gp_t[i]), float(gg_t[j]))
    return best


def main():
    op = pd.read_parquet('oof_pair.parquet')
    op_sorted = op.sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    yp = op_sorted.contact.values.astype(np.int8)
    ps = op_sorted.prob.values.astype(np.float64)

    mine = pd.read_parquet('oof_ground.parquet')[['contact_id','contact','prob']].rename(columns={'prob':'m_mine'})
    cast = pd.read_parquet('../castor_alpha/ground_oof_tab.parquet')[['contact_id','oof']].rename(columns={'oof':'m_cast'})
    ach = pd.read_parquet('../achernar_sigma/oof_ground_lgbv1.parquet')[['contact_id','pred']].rename(columns={'pred':'m_ach'})
    d = mine.merge(cast, on='contact_id').merge(ach, on='contact_id')
    yg = d.contact.values.astype(np.int8)
    w = json.load(open('blend_ground3.json'))['ground_weights']
    gprob = w[0]*d.m_mine.values + w[1]*d.m_cast.values + w[2]*d.m_ach.values
    gdf = d[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']] if 'nfl_player_id_1' in d else None
    if gdf is None:
        tmp = pd.read_parquet('oof_ground.parquet')
        gdf = d[['contact_id']].merge(tmp[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']], on='contact_id')
    gdf = gdf.copy(); gdf['_p'] = gprob; gdf['_y'] = yg

    best = (-1, None)
    cache = {}
    for wp in [1,3,5,7,9,11,15,21]:
        if wp == 1:
            pd_ = op_sorted; ps_w = ps
        else:
            pd_, ps_w = smooth(op_sorted, ps, wp)
        ps_w = ps_w.astype(np.float64)
        for wg in [1,3,5,7,9,11,15,21]:
            if wg == 1:
                gd_ = gdf.sort_values(KEYS, kind='mergesort').reset_index(drop=True); pg_w = gd_['_p'].values.astype(np.float64)
            else:
                gd_, pg_w = smooth(gdf, gprob, wg); pg_w = pg_w.astype(np.float64)
            yg_w = gd_['_y'].values.astype(np.int8)
            r = joint_mcc(ps_w, yp, pg_w, yg_w, nq=140)
            print('wp',wp,'wg',wg,round(r[0],5),round(r[1],5),round(r[2],5), flush=True)
            if r[0] > best[0]:
                best = (r[0], (wp,wg,r[1],r[2]))
    print('BEST', best)
    json.dump({'joint_mcc': best[0], 'wp': best[1][0], 'wg': best[1][1],
               'thr_pair': best[1][2], 'thr_ground': best[1][3]}, open('postprocess_result.json','w'), indent=2)


if __name__ == '__main__':
    main()
