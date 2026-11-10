"""Compare paired teacher trajectories before and after first appointment using stored DAG streams."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import beta

from flow import FIELDS, field_local, mean_q
from motion import bootstrap
from native_dag import DIMENSIONS
from native_flow import groups_in, subtract

CUTOFF = 2025
K = 64
BAND = 0.10
SEED = 421110
METRICS = ('con', 'div', 'multi', 'branch', 'excess')
STAGES = ('pre', 'post')
SIGNALS = ('con', 'div', 'multi', 'branch')
DRAWS = 64
COMPARISON_STATES = ('insufficient_calendar','zero_both','zero_pre','zero_post',
                     'too_few_positions','insufficient_local_stream','paired')


def read_data(db_file):
    with sqlite3.connect(db_file.resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        roster = pd.read_sql_query(
            '''SELECT uid,field,appointment_year,appointment_basis
               FROM faculty_observation WHERE n_supported_refs>0 ORDER BY uid''', db)
        publications = pd.read_sql_query(
            'SELECT uid,work,year,field FROM publication_metadata ORDER BY uid,year,work', db)
        links = {}
        for uid, citing, cited in db.execute('SELECT uid,citing,cited FROM linked_earlier_citations'):
            links.setdefault((uid,citing),set()).add(cited)
    return roster, publications, links


def event_windows(event, first_year):
    width = min(5,(CUTOFF-event-2)//2,(event-first_year)//2)
    if width<2:
        return 0, {}
    return width, {'pre':(event-2*width,event-width-1,event-width,event-1),
                   'post':(event+3,event+2+width,event+3+width,event+2+2*width)}


def interval_mask(years, lo, hi):
    return (years>=lo)&(years<=hi)


def seen_topics(years, groups, event, width, target):
    """Earlier support using event-aligned blocks; year 0 and years 1–2 are separate."""
    prior = years<target
    ages = years[prior]-event
    cluster = groups[prior]
    block = np.where(ages<0, (-ages-1)//width, np.where(ages==0,-1,np.where(ages<=2,-2,-3-(ages-3)//width)))
    seen = set()
    for b in np.unique(block):
        labels = cluster[block==b]
        n = np.bincount(labels,minlength=K)
        seen.update(np.flatnonzero((n>=2)&(n>=.15*len(labels))))
    return seen


def registry_from_stream(path, dimension):
    with np.load(path,allow_pickle=False) as stream:
        years = stream['year']
        n = stream['n']
        center = stream['center']
        fields = stream['field']
    assert tuple(fields)==FIELDS and center.shape==(*n.shape,dimension)
    assert np.array_equal(years,np.arange(years[0],CUTOFF+1))
    sums = np.nan_to_num(center).astype(np.float64)*n[...,None]
    def prefix(a):
        return np.concatenate([np.zeros_like(a[:1]),a.cumsum(0)],axis=0)
    return int(years[0]), {str(f):(prefix(n[i]),prefix(sums[i])) for i,f in enumerate(fields)}


def describe_phase(year, work, cluster, coords, linked_year, event, width, bounds, direct, peer, peer_own, origin, clouds):
    first, split, start, end = bounds
    ai,bi = interval_mask(year,first,split),interval_mask(year,start,end)
    a,b = coords[ai],coords[bi]
    row = {'source_linked':int(interval_mask(linked_year,first,split).sum()),
           'target_linked':int(interval_mask(linked_year,start,end).sum()),
           'source_positioned':len(a),'target_positioned':len(b),
           **{m:np.nan for m in METRICS},'branch_opportunity':np.nan,'supported_mass':np.nan}
    if len(b)<2:
        return row
    gb, separation = groups_in(cluster[bi],b,K)
    row['multi']=int(separation>=1)
    if len(a)<2:
        return row
    ga, _ = groups_in(cluster[ai],a,K)
    new = gb-seen_topics(year,cluster,event,width,start)
    retained = ga&gb
    retained_works = set(work[ai&np.isin(cluster,tuple(retained))])
    novel_works = work[bi&np.isin(cluster,tuple(new))]
    row.update(branch=int(any(retained_works&direct.get(w,set()) for w in novel_works)),
               branch_opportunity=int(bool(retained and new)))
    aa = subtract(peer,*peer_own,first,split,origin,K)
    bb = subtract(peer,*peer_own,start,end,origin,K)
    mass = np.bincount(cluster[ai],minlength=K)/len(a)
    _,_,support,shifts = field_local(aa,bb,mass,K)
    row['supported_mass']=support
    if shifts is None:
        return row
    advected = a+shifts[cluster[ai]]
    q_actual,q_passive = mean_q(b.astype(float)),mean_q(advected)
    if clouds is not None:
        clouds.append((advected,b))
    if q_actual+q_passive>1e-12:
        excess=(q_actual-q_passive)/(q_actual+q_passive)
        row.update(excess=excess,con=int(excess<=-BAND),div=int(excess>=BAND))
    return row


def classify(row):
    if not row['width']:
        return 'insufficient_calendar'
    pre = row['pre_source_linked']+row['pre_target_linked']
    post = row['post_source_linked']+row['post_target_linked']
    if not pre or not post:
        return 'zero_both' if not pre and not post else 'zero_pre' if not pre else 'zero_post'
    if min(row[f'{stage}_{window}_positioned'] for stage in STAGES for window in ('source','target'))<2:
        return 'too_few_positions'
    if not np.isfinite(row['pre_excess']) or not np.isfinite(row['post_excess']):
        return 'insufficient_local_stream'
    return 'paired'


def match_volume(clouds, rng):
    """Keep the full-window background shifts and thin corresponding clouds equally."""
    estimates = []
    sizes = np.min([[len(a),len(b)] for a,b in clouds],axis=0)

    def q_sample(x, n):
        if len(x)==n:
            return np.full(DRAWS,mean_q(x.astype(float)))
        ids = np.stack([rng.choice(len(x),n,replace=False) for _ in range(DRAWS)])
        data = x[ids].astype(float,copy=False)
        centered = data-data.mean(axis=1,keepdims=True)
        return 2*np.einsum('ijk,ijk->i',centered,centered)/(n-1)

    for a,b in clouds:
        qa,qb = q_sample(a,sizes[0]),q_sample(b,sizes[1])
        e = np.divide(qb-qa,qa+qb,out=np.full_like(qa,np.nan),where=qa+qb>1e-12)
        ok = np.isfinite(e)
        if not ok.any():
            return {}
        e = e[ok]
        estimates.append((np.mean(e<=-BAND),np.mean(e>=BAND),e.mean()))
    return {f'{stage}_thinned_{metric}':float(x) for stage,est in zip(STAGES,estimates,strict=True)
            for metric,x in zip(('con','div','excess'),est,strict=True)}


def exact_difference(gain, loss, n):
    """95% simultaneous bound for paired proportions from two exact binomial intervals."""
    tail = 0.05/4
    def interval(k):
        return (float(beta.ppf(tail,k,n-k+1)) if k else 0.,
                float(beta.ppf(1-tail,k+1,n-k)) if k<n else 1.)
    a,b = interval(gain),interval(loss)
    return [(gain-loss)/n,a[0]-b[1],a[1]-b[0]]


def rows_for_dimension(roster, publications, links, folder, dimension):
    axis=np.load(folder/'paper_axis.npz',allow_pickle=False)
    vectors=np.load(folder/'paper_vectors.npy',mmap_mode='r',allow_pickle=False)
    assert vectors.shape==(len(axis['work']),dimension)
    work_index=pd.Series(np.arange(len(axis['work'])),index=axis['work'])
    clusters=axis['cluster']
    origin,peer=registry_from_stream(folder/'annual_stream.npz',dimension)
    publications=publications.assign(ix=publications.work.map(work_index))
    unique = publications.dropna(subset=['ix']).sort_values(['year','work','uid']).drop_duplicates(['field','work'])
    work_registry = {f:g.set_index('work') for f,g in unique.groupby('field')}
    for field, registry in work_registry.items():
        expected = peer[field][0][-1,-1]
        assert expected==len(registry),(field,expected,len(registry))
    by_uid={u:g for u,g in publications.groupby('uid',sort=False)}
    rows=[]
    rng=np.random.default_rng(SEED)
    for r in roster.itertuples(index=False):
        if pd.isna(r.appointment_year):
            continue
        event=int(r.appointment_year)
        g=by_uid[r.uid]
        pos=g.dropna(subset=['ix'])
        year=pos.year.to_numpy()
        work=pos.work.to_numpy()
        ix=pos.ix.to_numpy(dtype=int)
        label=clusters[ix]
        linked_year=g.year.to_numpy()
        early_n=int(interval_mask(linked_year,event+1,event+2).sum())
        late_n=int((linked_year>=event+3).sum())
        early_years=max(0,min(2,CUTOFF-event))
        late_years=max(0,CUTOFF-event-2)
        early_mask=interval_mask(year,event+1,event+2)
        early_geo=int(early_mask.sum())
        early_multi=np.nan
        if early_years==2 and early_geo>=2:
            _,score=groups_in(label[early_mask],vectors[ix[early_mask]],K)
            early_multi=int(score>=1)
        width,windows=event_windows(event,origin)
        row={'uid':r.uid,'field':r.field,'event':event,'basis':r.appointment_basis,'width':width,
             'ever_pre':int((linked_year<event).sum()),'year_zero':int((linked_year==event).sum()),
             'early_years':early_years,'early_n':early_n,'early_geo':early_geo,'early_multi':early_multi,
             'late_years':late_years,'late_n':late_n}
        clouds=[] if dimension==1024 else None
        if width:
            own = work_registry[r.field].loc[pos.work.unique()]
            own_ix = own.ix.to_numpy(dtype=int)
            peer_own = (own.year.to_numpy(),clusters[own_ix],vectors[own_ix])
            citations = {w:links.get((r.uid,w),set()) for w in work}
            for stage,bounds in windows.items():
                assert bounds[0]>=origin and bounds[-1]<=CUTOFF
                values=describe_phase(year,work,label,vectors[ix],linked_year,event,width,bounds,
                                      citations,peer[r.field],peer_own,origin,clouds)
                row.update({f'{stage}_{key}':v for key,v in values.items()})
        row['comparison']=classify(row)
        if clouds is not None and row['comparison']=='paired':
            assert len(clouds)==2
            row.update(match_volume(clouds,rng))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_coverage(roster, rows):
    out={}
    for f in FIELDS:
        base=roster if f=='all' else roster[roster.field==f]
        sub=rows if f=='all' else rows[rows.field==f]
        early_full=sub.early_years.eq(2)
        late_mature=sub.late_years.ge(3)
        paired=sub.comparison.eq('paired')
        counts={s:int(sub.comparison.eq(s).sum()) for s in COMPARISON_STATES}
        assert sum(counts.values())==len(sub)
        out[f]={'base':len(base),'dated':len(sub),'date_unresolved':len(base)-len(sub),
                'ever_pre_zero':int(sub.ever_pre.eq(0).sum()),
                'early_full':int(early_full.sum()),'early_full_zero':int((early_full&sub.early_n.eq(0)).sum()),
                'early_short':int((~early_full).sum()),
                'early_short_zero':int((~early_full&sub.early_n.eq(0)).sum()),
                'early_multi_eligible':int(sub.early_multi.notna().sum()),
                'early_multi':int(sub.early_multi.sum()),
                'late_mature':int(late_mature.sum()),'late_mature_zero':int((late_mature&sub.late_n.eq(0)).sum()),
                'late_not_started':int(sub.late_years.eq(0).sum()),
                'late_1_or_2_years':int(sub.late_years.between(1,2).sum()),
                'late_1_or_2_zero':int((sub.late_years.between(1,2)&sub.late_n.eq(0)).sum()),
                'states':counts,'paired_widths':{str(w):int((paired&sub.width.eq(w)).sum()) for w in (2,3,4,5)}}
        assert out[f]['late_mature']+out[f]['late_not_started']+out[f]['late_1_or_2_years']==len(sub)
    return out


def summarize(rows):
    output={}
    rng=np.random.default_rng(SEED)
    for field in FIELDS:
        s=rows if field=='all' else rows[rows.field==field]
        out={}
        for group in ('matched','five_year'):
            keep=s if group=='matched' else s[s.width==5]
            old=keep[[f'pre_{m}' for m in METRICS]].to_numpy()
            new=keep[[f'post_{m}' for m in METRICS]].to_numpy()
            assert np.isfinite(old).all() and np.isfinite(new).all()
            n=len(keep)
            interval=bootstrap(new-old,rng) if n else np.full((len(METRICS),3),np.nan)
            for i,m in enumerate(SIGNALS):
                if n:
                    gain = int(((old[:,i]==0)&(new[:,i]==1)).sum())
                    loss = int(((old[:,i]==1)&(new[:,i]==0)).sum())
                    interval[i] = exact_difference(gain,loss,n)
            out[group]={'n':n,'metrics':{m:{'before':float(old[:,i].mean()) if n else None,
                       'after':float(new[:,i].mean()) if n else None,
                       'delta':interval[i].tolist() if n else None,
                       **({'before_count':int(old[:,i].sum()),'after_count':int(new[:,i].sum()),
                           'gain':int(((old[:,i]==0)&(new[:,i]==1)).sum()),
                           'loss':int(((old[:,i]==1)&(new[:,i]==0)).sum())} if m in SIGNALS else {})}
                       for i,m in enumerate(METRICS)},
                       'branch_opportunities':[int(keep[f'{p}_branch_opportunity'].sum()) for p in STAGES]}
        output[field]=out
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('database','tensors','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    roster,pubs,links=read_data(a.database)
    measured={}
    for d in DIMENSIONS:
        rows=rows_for_dimension(roster,pubs,links,a.tensors/f'd{d:04}',d)
        assert rows.uid.is_unique
        measured[d]=rows
        print(d,len(rows),int(rows.comparison.eq('paired').sum()),flush=True)
    common=set.intersection(*(set(r.loc[r.comparison.eq('paired'),'uid']) for r in measured.values()))
    for d,r in measured.items():
        r.loc[r.comparison.eq('paired')&~r.uid.isin(common),'comparison']='insufficient_local_stream'
        r.to_csv(a.output/f'faculty-d{d:04}.csv',index=False)
    additional={}
    rng=np.random.default_rng(SEED)
    for f in FIELDS:
        s=measured[1024] if f=='all' else measured[1024][measured[1024].field==f]
        additional[f]={}
        for m in ('multi','branch'):
            pairs=s[[f'pre_{m}',f'post_{m}']].dropna().to_numpy()
            n=len(pairs)
            gain=int(((pairs[:,0]==0)&(pairs[:,1]==1)).sum())
            loss=int(((pairs[:,0]==1)&(pairs[:,1]==0)).sum())
            additional[f][m]={'n':n,'before':float(pairs[:,0].mean()) if n else None,
                               'after':float(pairs[:,1].mean()) if n else None,
                               'delta':exact_difference(gain,loss,n) if n else None}
        matched=s[s.uid.isin(common)]
        additional[f]['position_medians']=[float(matched[f'{stage}_{window}_positioned'].median())
                                           for stage in STAGES for window in ('source','target')] if len(matched) else None
        metrics=('con','div','excess')
        pre=matched[[f'pre_thinned_{m}' for m in metrics]].to_numpy()
        post=matched[[f'post_thinned_{m}' for m in metrics]].to_numpy()
        ok=np.isfinite(pre-post).all(1)
        additional[f]['volume_matched']={'n':int(ok.sum()),'draws':DRAWS,'metrics':{
            m:{'before':float(pre[ok,i].mean()),'after':float(post[ok,i].mean()),
               'delta':bootstrap(post[ok,i]-pre[ok,i],rng).tolist()} for i,m in enumerate(metrics)
        }} if ok.any() else {'n':0,'draws':DRAWS,'metrics':{}}
    result={'supplemental':additional,'cutoff':CUTOFF,'seed':SEED,'common_n':len(common),'coverage':aggregate_coverage(roster,measured[1024]),
            'dimensions':{d:summarize(r[r.uid.isin(common)]) for d,r in measured.items()}}
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
