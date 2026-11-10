"""Evaluate appointment trajectories through the common cutoff without an elapsed-years gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from appointment_dag import BAND, CUTOFF, K, SEED, interval_mask, read_data, registry_from_stream
from flow import FIELDS, field_local, mean_q
from motion import bootstrap
from native_dag import DIMENSIONS
from native_flow import groups_in, subtract

EXPOSURE = {'all': (0,999),'0':(0,0),'1–2':(1,2),'3–5':(3,5),'6+':(6,999)}


def annual_multi(year, groups, x):
    values=[]
    for y in np.unique(year):
        mask=year==y
        if mask.sum()>=2:
            _,separation=groups_in(groups[mask],x[mask],K)
            values.append(separation>=1)
    return len(values),float(np.mean(values)) if values else None


def event_metrics(event, first, year, work, group, x, refs, registry, own, origin):
    before=year<event
    after=year>event
    early=interval_mask(year,event+1,event+2)
    out={}
    for phase,mask in (('pre',before),('post',after),('early',early)):
        n,rate=annual_multi(year[mask],group[mask],x[mask])
        out[phase+'_multi_years']=n
        out[phase+'_multi']=rate
    if min(before.sum(),after.sum())<2:
        return out
    pre_topics,_=groups_in(group[before],x[before],K)
    post_topics,_=groups_in(group[after],x[after],K)
    retained=pre_topics&post_topics
    novel=post_topics-set(group[year<=event])
    anchors=set(work[before&np.isin(group,tuple(retained))])
    new=work[after&np.isin(group,tuple(novel))]
    out['branch_opportunity']=int(bool(retained and novel))
    out['branch']=int(any(anchors&refs.get(w,set()) for w in new))
    a=subtract(registry,*own,max(origin,first),event-1,origin,K)
    b=subtract(registry,*own,event+1,CUTOFF,origin,K)
    prior=np.bincount(group[before],minlength=K)/before.sum()
    _,_,support,shifts=field_local(a,b,prior,K)
    out['supported_mass']=support
    if shifts is None:
        return out
    observed=mean_q(x[after].astype(float))
    advected=mean_q(x[before]+shifts[group[before]])
    if observed+advected>1e-12:
        excess=(observed-advected)/(observed+advected)
        out.update(excess=excess,con=int(excess<=-BAND),div=int(excess>=BAND))
    return out


def measure(roster, publications, links, folder, dimension):
    with np.load(folder/'paper_axis.npz',allow_pickle=False) as paper:
        ids=paper['work']
        labels=paper['cluster']
    x=np.load(folder/'paper_vectors.npy',mmap_mode='r',allow_pickle=False)
    assert x.shape==(len(ids),dimension)
    index=pd.Series(np.arange(len(ids)),index=ids)
    origin,peer=registry_from_stream(folder/'annual_stream.npz',dimension)
    pubs=publications.assign(ix=publications.work.map(index))
    unique=pubs.dropna(subset=['ix']).sort_values(['year','work','uid']).drop_duplicates(['field','work'])
    registry={f:g.set_index('work') for f,g in unique.groupby('field')}
    assert all(len(g)==peer[f][0][-1,-1] for f,g in registry.items())
    by_uid={u:g for u,g in pubs.groupby('uid',sort=False)}
    rows=[]
    for r in roster.itertuples(index=False):
        if pd.isna(r.appointment_year):
            rows.append({'uid':r.uid,'field':r.field,'dated':False})
            continue
        event=int(r.appointment_year)
        g=by_uid[r.uid]
        pos=g.dropna(subset=['ix'])
        ix=pos.ix.to_numpy(dtype=int)
        work=pos.work.to_numpy()
        year=pos.year.to_numpy()
        own=registry[r.field].loc[pos.work.unique()]
        own_ix=own.ix.to_numpy(dtype=int)
        linked=g.year.to_numpy()
        early=interval_mask(linked,event+1,event+2)
        row={'uid':r.uid,'field':r.field,'dated':True,'event':event,'basis':r.appointment_basis,
             'followup':max(0,CUTOFF-event),'pre_papers':int((linked<event).sum()),
             'post_papers':int((linked>event).sum()),'event_papers':int((linked==event).sum()),
             'early_papers':int(early.sum()),'pre_positioned':int((year<event).sum()),
             'post_positioned':int((year>event).sum())}
        row.update(event_metrics(event,int(linked.min()),year,work,labels[ix],x[ix],
                                 {w:links.get((r.uid,w),set()) for w in work},peer[r.field],
                                 (own.year.to_numpy(),labels[own_ix],x[own_ix]),origin))
        rows.append(row)
    result=pd.DataFrame(rows)
    assert len(result)==len(roster) and result.uid.is_unique
    return result


def summarize(frame):
    result={}
    rng=np.random.default_rng(SEED)
    def scalar(a):
        a=a.dropna().to_numpy()
        return {'n':len(a),'estimate':bootstrap(a[:,None],rng)[0].tolist() if len(a) else None}
    for field in FIELDS:
        data=frame if field=='all' else frame[frame.field==field]
        result[field]={}
        for age,(lo,hi) in EXPOSURE.items():
            dated=data[data.dated&data.followup.between(lo,hi)]
            multi=dated.dropna(subset=['pre_multi','post_multi'])
            result[field][age]={'base':len(data) if age=='all' else len(dated),
                'dated':len(dated),'no_followup':int(dated.followup.eq(0).sum()),
                'any_pre_post':int((dated.pre_papers.gt(0)&dated.post_papers.gt(0)).sum()),
                'observed_post_zero':int((dated.followup.gt(0)&dated.post_papers.eq(0)).sum()),
                'gcc_pre_post':int((dated.pre_positioned.gt(0)&dated.post_positioned.gt(0)).sum()),
                'con':scalar(dated.con),'div':scalar(dated['div']),'excess':scalar(dated.excess),
                'multi':{'n':len(multi),'pre':float(multi.pre_multi.mean()) if len(multi) else None,
                         'post':float(multi.post_multi.mean()) if len(multi) else None,
                         'delta':scalar(multi.post_multi-multi.pre_multi)['estimate']},
                'branch':scalar(dated.branch),
                'branch_opportunities':int(dated.branch_opportunity.sum()),
                'early_multi':scalar(dated.early_multi)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('portfolio','tensors','output'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    roster,pubs,links=read_data(a.portfolio)
    result={}
    for d in DIMENSIONS:
        data=measure(roster,pubs,links,a.tensors/f'd{d:04}',d)
        data.to_csv(a.output/f'faculty-d{d:04}.csv',index=False)
        result[d]=summarize(data)
        print(d,{k:result[d]['all']['all'][k] for k in ('any_pre_post','con','div','multi','branch')},flush=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
