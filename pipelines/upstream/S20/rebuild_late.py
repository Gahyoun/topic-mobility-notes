"""Classify late publication activity after an appointment transition period."""
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=Path.home()/'work/topic-mobility-private/outputs'
prev=root/'S17_appointment_evidence_20261109'
evidence=json.loads((prev/'evidence-export.json').read_text())
prior=json.loads((prev/'summary.json').read_text())
previous_followup=json.loads((root/'S19_post_appointment_followup_20261109'/'summary.json').read_text())
overlay={r['uid']:r for r in evidence}
fields=('mathematics','physics','chemistry','biology')
final_year=2025

def readonly(path):
    return path.resolve().as_uri()+'?mode=ro&immutable=1'

with np.load(root/'S08_space/space_b2025.npz',allow_pickle=False) as saved:
    vectors=saved['vectors']
    coordinate_ids=set(saved['ids'][np.isfinite(vectors).all(axis=1)&(np.linalg.norm(vectors,axis=1)>0)].tolist())
    assert saved['dims']==128 and saved['channel']=='refs'

with sqlite3.connect(readonly(root/'S14_full_roster_20261109/cohort-2025.sqlite'),uri=True) as db:
    db.row_factory=sqlite3.Row
    db.execute('ATTACH DATABASE ? AS web',(readonly(Path.cwd()/'Data/professor-career-dashboard/.private/mijin_publish_20261105/web.sqlite'),))
    cohort=[dict(r) for r in db.execute("""
      SELECT r.professor_uid AS uid,r.subject AS field,r.n_supported_refs,r.shared_author_id,r.evidence_first_year,
          EXISTS (SELECT 1 FROM web.semester_snapshots s WHERE s.professor_uid=r.professor_uid AND s.term='2026-spring') AS spring
      FROM roster_coverage r
    """)]
    papers=defaultdict(list)
    for uid,work,year,review,has_refs in db.execute("""
      SELECT f.professor_uid,f.work_id,f.year,f.review_only,c.state='cached_with_refs'
      FROM faculty_publication f JOIN reference_coverage c USING(work_id)
      JOIN roster_coverage r USING(professor_uid)
      WHERE r.n_supported_refs>0
    """):
        papers[uid].append((year,bool(review),bool(has_refs),bool(has_refs) and work in coordinate_ids))

def event(r,mode):
    e=overlay.get(r['uid'],{})
    if e.get('kind')=='official':
        return e['year'],'official'
    if mode=='official':
        return None,None
    if r['evidence_first_year'] is not None:
        return r['evidence_first_year'],'high_confidence'
    if mode=='proxy' and e.get('kind')=='first_affiliated_paper':
        return e['year'],'first_affiliated_paper'
    return None,None

def evaluate(r,anchor):
    if r['shared_author_id']:
        return {'status':'hold_shared_id'}
    pre_refs=Counter()
    pre_coords=Counter()
    historical_post_refs=Counter()
    late_coords=Counter()
    early=Counter()
    late=Counter()
    observed_until=final_year-anchor
    for year,review,refs,coords in papers[r['uid']]:
        rel=year-anchor
        if not review and refs and -10<=rel<0:
            pre_refs[rel//2]+=1
            pre_coords[rel//2]+=int(coords)
        if not review and refs and 0<=rel<=9:
            historical_post_refs[rel//2]+=1
        if 1<=rel<=2:
            early['review' if review else 'papers']+=1
        if 3<=rel<=observed_until:
            late['review' if review else 'papers']+=1
            start=3+2*((rel-3)//2)
            if not review and coords and start+1<=min(9,observed_until):
                late_coords[start]+=1
    pre_ok=max(pre_refs.values(),default=0)>=2
    historical=pre_ok and max(historical_post_refs.values(),default=0)>=2
    if not pre_ok:
        return {'status':'hold_pre_window','historical':historical}
    pre_q=max(pre_coords.values(),default=0)>=2
    q_ready=pre_q and max(late_coords.values(),default=0)>=2
    if late['papers']:
        status='post_late_q_ready' if q_ready else 'post_late_q_unavailable'
    elif late['review']:
        status='post_late_review'
    elif observed_until<5:
        status='post_followup_short'
    else:
        status='post_long_no_link'
    early_zero=observed_until>=2 and not early['papers'] and not early['review']
    return {'status':status,'historical':historical,'early_zero':early_zero,
            'early_papers':early['papers'],'early_review':early['review'],
            'late_papers':late['papers'],'late_review':late['review'],
            'late_qualified_2yr_windows':sum(n>=2 for n in late_coords.values()),
            'last_relative_year':observed_until,'active_topic_pool':status!='post_long_no_link'}

detail=[]
summary={}
for mode in ('official','baseline','proxy'):
    selected={}
    for r in cohort:
        if not r['n_supported_refs']:
            continue
        year,basis=event(r,mode)
        if year is not None:
            selected[r['uid']]={**r,'year':year,'basis':basis,**evaluate(r,year)}
    detail.extend({'mode':mode,**r} for r in selected.values())
    summary[mode]={}
    for scope in ('spring','all'):
        summary[mode][scope]={}
        for field in fields:
            base=[r for r in cohort if r['n_supported_refs']>0 and r['field']==field and (scope=='all' or r['spring'])]
            rows=[selected[r['uid']] for r in base if r['uid'] in selected]
            counts=Counter(r['status'] for r in rows)
            outs={k:v for k,v in counts.items() if k.startswith('post_')}
            early_zero=[r for r in rows if r.get('early_zero')]
            assert len(early_zero)==previous_followup[mode][scope][field]['outcomes'].get('post_no_link',0)
            assert sum(outs.values())==previous_followup[mode][scope][field]['followed']
            old=prior[scope][field]
            assert old['baseline'][2]==len(base)
            if mode!='official':
                old_counts=old['baseline' if mode=='baseline' else 'counts']
                assert len(rows)==old_counts[3]
                assert sum(bool(r.get('historical')) for r in rows)==old_counts[4]
            summary[mode][scope][field]={
              'reference':len(base),'dated':len(rows),'followed':sum(outs.values()),
              'holds':{k:v for k,v in counts.items() if k.startswith('hold_')},
              'outcomes':outs,
              'early_zero_total':len(early_zero),
              'early_zero_resumed':sum(r['late_papers']>0 for r in early_zero),
              'early_zero_long':sum(r['status']=='post_long_no_link' for r in early_zero),
              'early_zero_review':sum(r['status']=='post_late_review' for r in early_zero),
              'early_zero_short':sum(r['status']=='post_followup_short' for r in early_zero),
              'early_paper_then_long':sum(r['status']=='post_long_no_link' and r['early_papers']>0 for r in rows),
              'active_topic_pool':sum(r.get('active_topic_pool',False) for r in rows),
            }
            cell=summary[mode][scope][field]
            assert cell['early_zero_total']==sum(cell[k] for k in ('early_zero_resumed','early_zero_long','early_zero_review','early_zero_short'))
            assert cell['active_topic_pool']==cell['followed']-outs.get('post_long_no_link',0)
            assert len(rows)==sum(counts.values())
assert len(cohort)==3937 and len(overlay)==583
(here/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(here/'records.json').write_text(json.dumps(detail,ensure_ascii=False)+'\n')
for mode,scopes in summary.items():
    for scope,data in scopes.items():
        print(mode,scope)
        for f,r in data.items(): print(f,r)
