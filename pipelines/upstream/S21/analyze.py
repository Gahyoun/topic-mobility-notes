"""Compare early-inclusive and later-only Q-hat within fixed faculty samples."""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=Path.home()/'work/topic-mobility-private/outputs'
s20=root/'S20_late_appointment_followup_20261109'
status=json.loads((s20/'records.json').read_text())
expected=json.loads((s20/'summary.json').read_text())
fields=('mathematics','physics','chemistry','biology')
pre=(-10,-8,-6,-4,-2)
late=(3,5,7)
cutoff=2025
ready=[r for r in status if r['status']=='post_late_q_ready']
uids={r['uid'] for r in ready}

with np.load(root/'S08_space/space_b2025.npz',allow_pickle=False) as saved:
    vectors=saved['vectors']
    valid=np.isfinite(vectors).all(axis=1)&(np.linalg.norm(vectors,axis=1)>0)
    positions={str(w):i for i,w in enumerate(saved['ids']) if valid[i]}
    assert saved['dims']==128 and saved['channel']=='refs'

uri=(root/'S14_full_roster_20261109/cohort-2025.sqlite').as_uri()+'?mode=ro&immutable=1'
with sqlite3.connect(uri,uri=True) as db:
    db.execute('CREATE TEMP TABLE needed (uid TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO needed VALUES (?)',((u,) for u in sorted(uids)))
    papers=defaultdict(lambda:defaultdict(list))
    for uid,work,year in db.execute("""
        SELECT n.uid,f.work_id,f.year
        FROM needed n JOIN faculty_publication f ON f.professor_uid=n.uid
        JOIN reference_coverage c USING(work_id)
        WHERE f.review_only=0 AND c.state='cached_with_refs'
        ORDER BY n.uid,f.year,f.work_id
    """):
        if work in positions:
            papers[uid][year].append(positions[work])

def qhat(inds):
    x=vectors[inds].astype(np.float64)
    n=len(x)
    summed=x.sum(axis=0)
    return float(max(0,2*(n*np.sum(x*x)-summed@summed)/(n*(n-1))))

def q_windows(uid,anchor,starts):
    by_year=papers[uid]
    out={}
    for t in starts:
        y=anchor+t
        if y+1>cutoff:
            continue
        indices=by_year.get(y,[])+by_year.get(y+1,[])
        if len(indices)>=2:
            out[t]=qhat(indices)
    return out

aligned={}
detail=[]
for r in ready:
    key=(r['uid'],r['year'])
    if key not in aligned:
        p=q_windows(*key,pre)
        b=q_windows(*key,late)
        e=q_windows(*key,(1,)).get(1)
        assert p and b and len(b)==r['late_qualified_2yr_windows']
        prev=float(np.mean(list(p.values())))
        later=float(np.mean(list(b.values())))
        inclusive=later if e is None else (e+sum(b.values()))/(len(b)+1)
        aligned[key]={'pre':prev,'early':e,'late':later,'inclusive':inclusive,
                      'first_late':b.get(3),'n_pre':len(p),'n_late':len(b)}
    detail.append({k:r[k] for k in ('mode','uid','field','spring','year','basis')}|aligned[key])

assert len(detail)==len({(r['mode'],r['uid']) for r in detail})
mode_summary={}
for mode in ('official','baseline','proxy'):
    mode_summary[mode]={}
    for scope in ('spring','all'):
        rows=[r for r in detail if r['mode']==mode and (scope=='all' or r['spring'])]
        for f in fields:
            assert sum(r['field']==f for r in rows)==expected[mode][scope][f]['outcomes'].get('post_late_q_ready',0)

        def summarize(sample, first=False):
            if not sample:
                return {'n':0}
            pre_level=np.array([r['pre'] for r in sample])
            early=np.array([r['early'] if r['early'] is not None else np.nan for r in sample])
            later=np.array([r['first_late' if first else 'late'] for r in sample])
            inclusive=(early+later)/2 if first else np.array([r['inclusive'] for r in sample])
            a,b=inclusive-pre_level,later-pre_level
            d=b-a
            matrix=np.column_stack([a,b,d])
            group=np.array([r['field'] for r in sample])
            indices=[np.flatnonzero(group==f) for f in fields if np.any(group==f)]
            ci=None
            if len(sample)>=5:
                rng=np.random.default_rng(20261109)
                values=[]
                for _ in range(10):
                    picks=np.concatenate([rng.choice(i,(1000,len(i)),replace=True) for i in indices],axis=1)
                    values.append(matrix[picks].mean(axis=1))
                ci=np.quantile(np.concatenate(values),[.025,.975],axis=0).T.tolist()
            both=np.isfinite(early)
            return {'n':len(sample),'early_valid':int(both.sum()),
                    'pre_mean':float(pre_level.mean()),
                    'post_early_mean':float(early[both].mean()) if both.any() else None,
                    'post_late_mean':float(later.mean()),
                    'delta_inclusive':float(a.mean()),'delta_late':float(b.mean()),
                    'difference':float(d.mean()),'ci':ci,
                    'sign_flips':int(((a*b<0)&(np.abs(a)>1e-12)&(np.abs(b)>1e-12)).sum()),
                    'mean_late_windows':float(np.mean([r['n_late'] for r in sample]))}

        panels={'common':([r for r in rows if r['early'] is not None],False),
                'all_late_ready':(rows,False),
                'common_first_late':([r for r in rows if r['early'] is not None and r['first_late'] is not None],True)}
        mode_summary[mode][scope]={}
        for name,(sample,first) in panels.items():
            mode_summary[mode][scope][name]={
                **{f:summarize([r for r in sample if r['field']==f],first) for f in fields},
                'all':summarize(sample,first),
            }

assert all(r['early'] is not None or r['late']==r['inclusive'] for r in detail)
inds=next(inds[:3] for ys in papers.values() for inds in ys.values() if len(inds)>=3)
x=vectors[inds].astype(np.float64)
assert abs(qhat(inds)-np.mean([np.sum((x[i]-x[j])**2) for i in range(3) for j in range(3) if i!=j]))<1e-12
(here/'summary.json').write_text(json.dumps(mode_summary,ensure_ascii=False,indent=2)+'\n')
(here/'paired_records.json').write_text(json.dumps(detail,ensure_ascii=False)+'\n')
for mode in ('official','baseline','proxy'):
    print(mode)
    for subset,data in mode_summary[mode]['spring'].items():
        print(subset)
        for f,r in data.items():
            print(f,r.get('n'),*(round(r[k],4) for k in ('delta_inclusive','delta_late','difference') if k in r),
                  [round(x,4) for x in r['ci'][2]] if r.get('ci') else None)
