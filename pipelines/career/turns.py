"""Analyze ordered topic motion and audit losses on the fixed prior paper inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import aggregation
from analyze import FIELDS, ARCHIVE_YEAR
from motion import WIDTHS, bootstrap, centroid
from window_support import block_labels, fixed_papers

EPS = 1e-12
METRICS = ('cosine','turn90','continuation','orthogonal','diff_diff','conv_conv',
           'diff_conv','conv_diff','neutral','switch','dot_rate','dot_rate_adjusted')
REASONS = ('no_early','short_horizon','sparse','gap','zero','eligible')


def score_triple(a,b,c,anchor):
    x,tx,_=a
    y,ty,noise=b
    z,tz,_=c
    first=y-x
    second=z-y
    length=np.linalg.norm(first)*np.linalg.norm(second)
    if min(np.linalg.norm(first),np.linalg.norm(second)) <= EPS:
        return None
    dot=first@second
    cosine=float(np.clip(dot/length,-1.,1.))
    elapsed=(ty-tx)*(tz-ty)
    assert elapsed>0
    direction=(np.nan,)*6
    if anchor is not None:
        r=np.linalg.norm(np.stack((x,y,z))-anchor,axis=1)
        u,v=np.where(abs(np.diff(r))<=EPS,0.,np.sign(np.diff(r)))
        direction=(u==v==1,u==v==-1,u==1 and v==-1,u==-1 and v==1,
                   u==0 or v==0,u*v==-1)
    return np.array((cosine,cosine < -EPS,cosine > EPS,abs(cosine)<=EPS,
                     *direction,dot/elapsed,(dot+noise)/elapsed),dtype=float)


def evaluate(x,year,max_year,anchor,width):
    bins=block_labels(year,width,'separate_early')
    windows={}
    for j in np.unique(bins[bins>=0]):
        sel=bins==j
        if sel.sum()>=2:
            windows[j]=centroid(x[sel],year[sel])
    pair=any(j+1 in windows for j in windows)
    triples=[]
    for j in sorted(windows):
        if j+1 in windows and j+2 in windows:
            score=score_triple(windows[j],windows[j+1],windows[j+2],anchor)
            if score is not None:
                triples.append(score)
    if max_year < 3+2*width:
        assert not triples and len(windows)<3
        failure='short_horizon'
    elif len(windows)<3:
        failure='sparse'
    elif not any(j+1 in windows and j+2 in windows for j in windows):
        failure='gap'
    elif not triples:
        failure='zero'
    else:
        failure='eligible'
    v=np.mean(triples,axis=0) if triples else np.full(len(METRICS),np.nan)
    if triples and anchor is not None:
        np.testing.assert_allclose(v[1:4].sum(),1,atol=1e-12)
        np.testing.assert_allclose(v[4:9].sum(),1,atol=1e-12)
    return ('no_early' if anchor is None else failure),v,len(triples),pair


def sample_summary(sample,fixed,centers,time_col,previous):
    groups=list(fixed.groupby(['uid','field'],sort=True))
    fields=np.array([key[1] for key,_ in groups])
    n=len(groups)
    results=np.full((n,3,len(METRICS)),np.nan)
    reasons=np.empty((n,3),object)
    triples=np.zeros((n,3),int)
    pairs=np.zeros((n,3),bool)
    paper_counts=np.zeros(n,int)
    spans=np.zeros(n,int)
    for i,(_,works) in enumerate(groups):
        year=works[time_col].to_numpy()
        anchor_year=works.anchor.iloc[0]
        assert works.anchor.nunique()==1
        max_year=(2025-anchor_year)//2*2 if sample=='database' else min(8,(ARCHIVE_YEAR-anchor_year-1)//2*2)
        assert year.max()<=max_year
        x=centers[works.ix.to_numpy()]
        anchor=x[year<3].mean(axis=0) if (year<3).sum()>=2 else None
        paper_counts[i]=len(year)
        spans[i]=year.max()
        for w,width in enumerate(WIDTHS):
            reasons[i,w],results[i,w],triples[i,w],pairs[i,w]=evaluate(x,year,max_year,anchor,width)
        if anchor is None:
            pairs[i,:]=False
    eligible=reasons=='eligible'
    common=eligible.all(1)
    old_common=pairs.all(1)
    assert (common<=old_common).all()
    rng=np.random.default_rng(20261110)
    out={}

    def characteristics(mask):
        return {'n':int(mask.sum()),'median_papers':float(np.median(paper_counts[mask])),
                'median_last_year':float(np.median(spans[mask]))} if mask.any() else {'n':0}

    for field in FIELDS:
        selected=np.ones(n,dtype=bool) if field=='all' else fields==field
        base=previous[field]
        old=selected & old_common
        m=selected & common
        assert selected.sum()==base['base'] and old.sum()==base['common']
        assert (selected & (reasons[:,0]!='no_early')).sum()==base['early']
        common_stats=bootstrap(results[m],rng) if m.sum()>=10 else None
        diff_stats=bootstrap(results[m,1:]-results[m,:1],rng) if m.sum()>=10 else None
        row={'base':int(selected.sum()),'previous_common':int(old.sum()),'common':int(m.sum()),
             'common_excluded':characteristics(selected & ~common),
             'common_included':characteristics(m),
             'previous_common_lost':int((old & ~common).sum()),
             'n_widths':dict(Counter(eligible[selected].sum(axis=1).astype(str))),
             'widths':{}}
        for w,width in enumerate(WIDTHS):
            avail=selected & eligible[:,w]
            excluded=selected & ~eligible[:,w]
            counts={r:int((selected & (reasons[:,w]==r)).sum()) for r in REASONS}
            assert sum(counts.values())==row['base']
            assert (selected & pairs[:,w]).sum()==base['widths'][str(width)]['n_available']
            all_estimate=bootstrap(results[avail,w],rng) if avail.sum()>=10 else None
            row['widths'][str(width)]={
                'available':int(avail.sum()),'triple_count':int(triples[avail,w].sum()),
                'previous_common_available':int((old & eligible[:,w]).sum()),
                'angle_only_no_anchor':int((selected & (reasons[:,w]=='no_early') & (triples[:,w]>0)).sum()),
                'common_triple_count':int(triples[m,w].sum()),'reasons':counts,
                'available_estimate':dict(zip(METRICS,all_estimate.tolist())) if all_estimate is not None else None,
                'common_estimate':dict(zip(METRICS,common_stats[w].tolist())) if common_stats is not None else None,
                'common_vs_2yr':dict(zip(METRICS,diff_stats[w-1].tolist())) if w and diff_stats is not None else None,
                'excluded_profile':characteristics(excluded),'available_profile':characteristics(avail),
            }
        out[field]=row
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','appointments','pilot_reference','pilot_rebin','cohort','appointment_evidence',
                'career_reference','motion_reference','out'):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    previous=json.loads(args.motion_reference.read_text())
    result={}

    def collect(sample,rows,windows,centers,time_col):
        result[sample]=sample_summary(sample,fixed_papers(rows,windows),centers,time_col,previous[sample])

    sources=('archive','appointments','pilot_reference','pilot_rebin','cohort','appointment_evidence',
             'career_reference')
    aggregation.main([str(getattr(args,k)) for k in sources]+
                     [str(args.out.with_name('aggregation-reproduced.json'))],collect=collect)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for sample,fields in result.items():
        for field,row in fields.items():
            print(sample,field,row['base'],row['previous_common'],row['common'],row['n_widths'])
            for width,item in row['widths'].items():
                print(width,item['reasons'],item['angle_only_no_anchor'],
                      item['triple_count'],item['common_triple_count'],
                      {k:round(v[0],4) for k,v in (item['common_estimate'] or item['available_estimate'] or {}).items()})


if __name__=='__main__':
    main()
