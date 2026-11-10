"""Estimate yearly peer streams and five-year portfolio transport in a native DAG space."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from flow import FIELDS, MIN_PEERS, field_local, mean_q, summarize_registry
from motion import bootstrap
from native_dag import DIMENSIONS


def subtract(registry, years, groups, vectors, lo, hi, origin, k):
    n = registry[0][hi-origin+1] - registry[0][lo-origin]
    s = registry[1][hi-origin+1] - registry[1][lo-origin]
    keep = (years >= lo) & (years <= hi)
    np.add.at(n, groups[keep], -1)
    np.add.at(s, groups[keep], -vectors[keep])
    n[k] -= keep.sum()
    s[k] -= vectors[keep].sum(0)
    assert (n >= 0).all()
    return n, s


def groups_in(labels, cloud, k):
    n = np.bincount(labels, minlength=k)
    present = np.flatnonzero((n >= 2) & (n >= .15 * len(labels)))
    centers = [cloud[labels == g].mean(0) for g in present]
    variances = [float(np.square(cloud[labels == g] - m).sum(1).mean())
                 for g, m in zip(present, centers, strict=True)]
    score = max((np.linalg.norm(a-b)/np.sqrt(max(variances[i]+variances[j],1e-12))
                 for i, a in enumerate(centers) for j, b in enumerate(centers[:i])), default=0.0)
    return set(present), float(score)


def run(database, output, dimension):
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        work, groups = zip(*db.execute('SELECT work,cluster FROM dag_native_coordinates ORDER BY work'))
        n = len(work)
        x = np.empty((n, dimension), dtype=np.float32)
        for i, (vector,) in enumerate(db.execute('SELECT vector FROM dag_native_coordinates ORDER BY work')):
            x[i] = np.frombuffer(vector, dtype='<f4', count=dimension)
        frame = pd.read_sql_query('SELECT uid,work,year,field FROM publication_metadata', db)
        windows = pd.read_sql_query('SELECT uid,field,year_start,year_end,n_papers,exposure '
                                    'FROM portfolio_windows ORDER BY uid,year_start', db)
        links = db.execute('SELECT uid,citing,cited FROM linked_earlier_citations')
        direct = {}
        for uid, citing, cited in links:
            direct.setdefault((uid, citing), set()).add(cited)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(output/'paper_vectors.npy', x, allow_pickle=False)
    np.savez_compressed(output/'paper_axis.npz', work=np.array(work), cluster=np.array(groups))
    frame['ix'] = frame.work.map(pd.Series(np.arange(n), index=work))
    frame = frame.dropna(subset=['ix']).astype({'ix':'int32'})
    frame['cluster'] = np.asarray(groups, dtype=int)[frame.ix]
    frame = frame.sort_values(['year','work','uid'])
    registry_all = frame.drop_duplicates('work')
    registry_field = frame.drop_duplicates(['field','work'])
    origin = int(frame.year.min())
    years = np.arange(origin, 2026, dtype=int)
    k = 64
    reg = {'all':summarize_registry(registry_all,x,origin,len(years),k)}
    reg.update({f:summarize_registry(g,x,origin,len(years),k) for f,g in registry_field.groupby('field')})
    field_works = {f:g.set_index('work') for f,g in registry_field.groupby('field')}
    global_works = registry_all.set_index('work')
    corpus_counts = np.stack([np.diff(reg[f][0],axis=0) for f in FIELDS])
    sums = np.stack([np.diff(reg[f][1],axis=0) for f in FIELDS])
    centers = np.divide(sums,corpus_counts[...,None],out=np.full_like(sums,np.nan),
                        where=corpus_counts[...,None]>0)
    velocity = np.diff(centers,axis=1)
    annual_support = (corpus_counts[:,1:]>=MIN_PEERS)&(corpus_counts[:,:-1]>=MIN_PEERS)
    velocity[~annual_support] = np.nan
    np.savez_compressed(output/'annual_stream.npz', field=FIELDS, year=years, cluster=np.arange(k+1),
                        n=corpus_counts, center=centers.astype('f4'), target_year=years[1:],
                        velocity=velocity.astype('f4'), supported=annual_support)
    # Project for display only. The basis is fixed across all years in this dimension.
    fit = centers[0,:,:k][corpus_counts[0,:,:k]>=MIN_PEERS]
    mean = fit.mean(0)
    from sklearn.utils.extmath import randomized_svd
    _, _, basis = randomized_svd(fit-mean, n_components=2, random_state=321110)
    display_center = (centers-mean)@basis.T
    display_velocity = velocity@basis.T
    np.savez_compressed(output/'display_projection.npz', mean=mean, basis=basis)
    view = {}
    for f, name in enumerate(FIELDS):
        layer = {}
        for j,y in enumerate(years[1:]):
            good = annual_support[f,j,:k]
            if good.any():
                layer[int(y)] = np.round(np.column_stack((
                    display_center[f,j,:k][good], display_velocity[f,j,:k][good],
                    corpus_counts[f,j,:k][good], np.flatnonzero(good))),6).tolist()
        view[name] = layer
    (output/'stream_display.json').write_text(json.dumps(view,separators=(',',':'))+'\n')
    by_uid = {u:g for u,g in frame.groupby('uid',sort=False)}
    movement, spread, vectors, observed = [], [], [], []
    for uid, parts in windows.groupby('uid',sort=True):
        field = parts.field.iloc[0]
        g = by_uid.get(uid,frame.iloc[:0])
        gy, gc, gx, gw = g.year.to_numpy(), g.cluster.to_numpy(), x[g.ix], g.work.to_numpy()
        focal = g.work.unique()
        peers = []
        for registry in (field_works[field],global_works):
            own = registry.loc[focal]
            peers.append((own.year.to_numpy(),own.cluster.to_numpy(),x[own.ix]))
        previous = None
        seen = set()
        for window in parts.itertuples(index=False):
            selected = (gy >= window.year_start) & (gy <= window.year_end)
            b, bc, bw = gx[selected], gc[selected], gw[selected]
            n_b = len(b)
            state = ('zero' if not window.n_papers else 'unpositioned' if not n_b else
                     'singleton' if n_b==1 else 'multiple')
            gb, sep = groups_in(bc,b,k)
            observed.append((uid,field,window.year_start,window.year_end,window.n_papers,n_b,state,
                             sep if len(gb)>=2 else None,int(sep>=1) if n_b>=2 else None))
            if previous is not None:
                wp, a, ac, aw, ga, ya = previous
                if len(a)>=2 and n_b>=2 and wp.exposure==window.exposure==5:
                    dt = gy[selected].mean()-ya
                    dy = b.mean(0,dtype=np.float64)-a.mean(0,dtype=np.float64)
                    fa = subtract(reg[field],*peers[0],wp.year_start,wp.year_end,origin,k)
                    fb = subtract(reg[field],*peers[0],window.year_start,window.year_end,origin,k)
                    ca = subtract(reg['all'],*peers[1],wp.year_start,wp.year_end,origin,k)
                    cb = subtract(reg['all'],*peers[1],window.year_start,window.year_end,origin,k)
                    mass = np.bincount(ac,minlength=k)/len(a)
                    field_move, local_move, share, shifts = field_local(fa,fb,mass,k)
                    corpus_move = cb[1][k]/cb[0][k]-ca[1][k]/ca[0][k] if min(cb[0][k],ca[0][k])>=MIN_PEERS else None
                    novel, retained = gb-seen, gb&ga
                    old = set(aw[np.isin(ac,list(retained))])
                    eligible = bw[np.isin(bc,list(novel))]
                    branch = sum(bool(old & direct.get((uid,p),set())) for p in eligible)
                    q_b = mean_q(b.astype(np.float64))
                    q_passive = mean_q(a+shifts[ac]) if shifts is not None else np.nan
                    excess = (q_b-q_passive)/(q_b+q_passive) if q_b+q_passive>1e-12 else np.nan
                    spread.append((q_b,q_passive,excess,len(novel),len(retained),branch,len(eligible)))
                    steps = np.array([dy,*[v if v is not None else np.full(dimension,np.nan)
                                            for v in (corpus_move,field_move,local_move)]])/dt
                    vectors.append(np.concatenate([steps,steps[:1]-steps[1:]],axis=0).astype('f4'))
                    movement.append((uid,field,wp.year_start,window.year_start,dt,len(a),n_b,share))
            previous = (window,b,bc,bw,gb,gy[selected].mean() if n_b else None)
            seen.update(gb)
    index = pd.DataFrame(movement,columns=['uid','field','source_start','target_start','years','n_prior','n_later','supported_mass'])
    measures = pd.DataFrame(spread,columns=['q_observed','q_passive','excess','novel','retained','branch_papers','new_papers'])
    vector = np.asarray(vectors,dtype=np.float32)
    for i, col in enumerate(('observed_speed','corpus_speed','field_speed','local_speed','corpus_residual','field_residual','local_residual')):
        measures[col] = np.linalg.norm(vector[:,i],axis=1)
    local = vector[:,3]
    norm = np.linalg.norm(local,axis=1)
    parallel = np.divide(np.einsum('ij,ij->i',vector[:,0],local),norm,
                         out=np.full_like(norm,np.nan),where=norm>1e-12)
    measures['parallel'] = parallel
    measures['transverse'] = np.sqrt(np.maximum(measures.observed_speed**2-parallel**2,0))
    measures['con'] = np.where(measures.excess.notna(),measures.excess<=-.10,np.nan)
    measures['div'] = np.where(measures.excess.notna(),measures.excess>=.10,np.nan)
    measures['branch'] = (measures.branch_papers>0).astype(int)
    index = pd.concat([index,measures],axis=1)
    index.to_csv(output/'transition_index.csv',index=False)
    np.save(output/'trajectory_tensor.npy',vector,allow_pickle=False)
    observed = pd.DataFrame(observed,columns=['uid','field','start','end','papers','positioned','state','separation','multi'])
    observed.to_csv(output/'window_observation.csv',index=False)
    stats = {}
    rng = np.random.default_rng(321110)
    metrics = ['excess','observed_speed','local_speed','local_residual','parallel','transverse','con','div']
    for f in FIELDS:
        sub = index if f=='all' else index[index.field==f]
        win = observed if f=='all' else observed[observed.field==f]
        win = win[(win.end-win.start)==4]
        support = sub.dropna(subset=['excess'])
        average = support.groupby('uid')[metrics].mean()
        interval = bootstrap(average.to_numpy(),rng) if len(average) else np.full((len(metrics),3),np.nan)
        stats[f] = {'transitions':len(sub),'faculty':int(sub.uid.nunique()),'local_pairs':len(support),
                    'local_faculty':len(average),'con_pairs':int(support.con.sum()),'div_pairs':int(support['div'].sum()),
                    'branch_pairs':int(sub.branch.sum()),'branch_possible':int((sub.novel.gt(0)&sub.retained.gt(0)).sum()),
                    'multi_windows':int(win.multi.sum()),'multi_eligible':int(win.multi.notna().sum()),
                    'state_counts':win.state.value_counts().to_dict(),
                    'estimates':{m:[float(y) if np.isfinite(y) else None for y in v] for m,v in zip(metrics,interval,strict=True)}}
    (output/'summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(dimension,stats['all'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('database',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--dimension',type=int,choices=DIMENSIONS)
    a = p.parse_args()
    for d in (a.dimension,) if a.dimension else DIMENSIONS:
        run(a.database,a.output/f'd{d:04}',d)


if __name__=='__main__':
    main()
