"""Paired validation of institution-publication event time on official appointments."""
import gzip
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent
private = Path.home()/'work/topic-mobility-private'
s14 = private/'outputs/S14_full_roster_20261109'
s17 = private/'outputs/S17_appointment_evidence_20261109'
space_path = private/'outputs/S08_space/space_b2025.npz'
web_path = Path.cwd()/'Data/professor-career-dashboard/.private/mijin_publish_20261105/web.sqlite'

# Exact academic institutions from the official first-appointment evidence.
# Shared parent IDs are accepted for branch campuses only with branch text.
institutions = {
    'Elrazi College of Medical and Technological Sciences': None,
    'Ewha Womans University':'I138925566','이화여자대학교':'I138925566',
    'GIST':'I39534123','KAIST':'I157485424','POSTECH':'I123900574','UNIST':'I48566637',
    'Rutgers University':'I102322142','University of Georgia':'I165733156',
    'University of Iowa':'I126307644','University of Maryland':'I66946132','University of Michigan':'I27837315',
    '강원대학교':'I165507594','건국대학교 글로컬':'I24062138','경상국립대학교':'I189442560',
    '고려대학교':'I197347611','단국대학교':'I89015989','대구한의대학교':'I138826464',
    '부산대학교':'I4921948','서강대학교':'I148751991','서울대학교':'I139264467',
    '성균관대학교':'I848706','아주대학교':'I57664883','연세대학교':'I193775966',
    '울산대학교':'I40542001','전남대학교':'I111277659','한동대학교':'I113825674',
    '한양대학교 ERICA':'I4575257',
}
branch_tokens = {'건국대학교 글로컬':r'chungju|glocal|global campus|충주|글로컬',
                 '한양대학교 ERICA':r'ansan|\berica\b|안산|에리카'}
branch_raw = {'건국대학교 글로컬':'rawinst:ae942e68122c84de',
              '한양대학교 ERICA':'rawinst:e2e3cdc3cbeecaa9'}

def ro(path):
    return path.resolve().as_uri()+'?mode=ro&immutable=1'

def normalize_work(value):
    match = re.fullmatch(r'W\d+', (value or '').strip().split('/')[-1].upper())
    return match[0] if match else None

official={r['uid']:r for r in json.loads((s17/'evidence-export.json').read_text()) if r['kind']=='official'}
assert set(institutions)=={r['institution'] for r in official.values()}
assert len(official)==100
with sqlite3.connect(ro(s14/'cohort-2025.sqlite'),uri=True) as db:
    db.row_factory=sqlite3.Row
    db.execute('ATTACH DATABASE ? AS web',(ro(web_path),))
    db.execute('ATTACH DATABASE ? AS refs',(ro(private/'refs/work_references.sqlite'),))
    db.execute('CREATE TEMP TABLE audit(uid TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO audit VALUES (?)',((uid,) for uid in official))
    aliases={normalize_work(a):normalize_work(b) for a,b in db.execute('SELECT requested_id,canonical_id FROM refs.work_alias')
             if normalize_work(a) and normalize_work(b) and normalize_work(a)!=normalize_work(b)}
    def canonical(value):
        key=normalize_work(value)
        visited=set()
        while key in aliases:
            assert key not in visited
            visited.add(key)
            key=aliases[key]
        return key
    db.create_function('canonical_work',1,canonical,deterministic=True)
    campus=dict(db.execute('SELECT unit_id,campus_cluster_id FROM web.institution_units'))
    assert set(institutions.values())-{None} <= set(campus)
    book = {r['uid']:dict(r) for r in db.execute("""
      SELECT a.uid,p.phd_year,r.shared_author_id,
        EXISTS(SELECT 1 FROM web.semester_snapshots s WHERE s.professor_uid=a.uid AND s.term='2026-spring') AS spring
      FROM audit a JOIN web.professors p ON p.professor_uid=a.uid
      JOIN roster_coverage r ON r.professor_uid=a.uid
    """)}
    affiliations=defaultdict(list)
    for r in db.execute("""
      SELECT a.uid, f.work_id, f.year, u.institution_unit_id, u.campus_cluster_id, u.raw_affiliation_string
      FROM audit a JOIN web.raw_affiliation_units u ON u.professor_uid=a.uid
      JOIN faculty_publication f ON f.professor_uid=a.uid AND f.work_id=canonical_work(u.work_id)
      WHERE f.review_only=0 AND u.identity_decision='keep'
    """):
        affiliations[r['uid']].append(dict(r))
    records=list(db.execute("""
      SELECT a.uid,f.work_id,f.year
      FROM audit a JOIN faculty_publication f ON f.professor_uid=a.uid
      JOIN reference_coverage c USING(work_id)
      WHERE f.review_only=0 AND c.state='cached_with_refs'
      ORDER BY a.uid,f.year,f.work_id
    """))
    roster=defaultdict(list)
    for r in db.execute("""
        SELECT a.uid,s.institution_unit_id,s.institution_raw,s.year,s.term,s.rank
        FROM audit a JOIN web.semester_snapshots s ON s.professor_uid=a.uid
        ORDER BY s.year, CASE WHEN s.term LIKE '%-fall' THEN 1 ELSE 0 END
    """):
        roster[r['uid']].append(dict(r))
    phd=defaultdict(set)
    for r in db.execute("""
        SELECT a.uid,e.institution_unit_id FROM audit a JOIN web.education e ON e.professor_uid=a.uid
        WHERE e.degree_level='phd'
    """):
        phd[r['uid']].add(r['institution_unit_id'])

def first_work(rows, matches):
    return min((r for r in rows if matches(r)), key=lambda r:(r['year'],r['work_id']),default=None)

def official_match(label,r):
    unit=institutions[label]
    if unit is None:
        return bool(re.search(r'\bel[ -]?razi\b',r['raw_affiliation_string'] or '',re.I))
    common=r['institution_unit_id']==unit or r['campus_cluster_id']==campus[unit]
    if label in branch_tokens:
        return r['institution_unit_id']==branch_raw[label] or (
            common and bool(re.search(branch_tokens[label],r['raw_affiliation_string'] or '',re.I)))
    if label=='연세대학교':
        return common and r['institution_unit_id']!='I4405258769' and not re.search(
            r'\bmirae\b|\bwonju\b|원주|미래캠퍼스',r['raw_affiliation_string'] or '',re.I)
    return common

def operational_match(r,anchor):
    return r['institution_unit_id']==anchor or (
        campus.get(anchor) is not None and r['campus_cluster_id']==campus[anchor])

dates = []
for uid,r in official.items():
    row={'uid':uid,'field':r['field'],'spring':book[uid]['spring'],'official_year':r['year'],
         'official_institution':r['institution'],'shared_author_id':book[uid]['shared_author_id']}
    papers=affiliations[uid]
    first=first_work(papers,lambda p:official_match(r['institution'],p))
    row.update(paper_year=first['year'] if first else None,
               anchor_work=first['work_id'] if first else None,
               availability='ok' if first else ('no_supported_affiliations' if not papers else 'no_official_institution_paper'))
    obs=roster[uid]
    ranked=[s for s in obs if s['rank'] in ('assistant','associate','full')]
    if obs:
        earliest=(ranked or obs)[0]
        options={s['institution_unit_id'] for s in (ranked or obs) if s['term']==earliest['term']}
        places={campus.get(x) or x for x in options}
        unit=min(options)
        row['roster_unit']=unit
        row['roster_matches_official']=any(official_match(r['institution'],{
            'institution_unit_id':s['institution_unit_id'],
            'campus_cluster_id':campus[s['institution_unit_id']],
            'raw_affiliation_string':s['institution_raw']})
            for s in (ranked or obs) if s['term']==earliest['term'])
        if len(places)==1:
            op=first_work(papers,lambda p:operational_match(p,unit))
            if op:
                row['roster_paper_year']=op['year']
                degree=book[uid]['phd_year']
                same_phd=any(x==unit or (campus.get(x) and campus[x]==campus[unit]) for x in phd[uid])
                if not book[uid]['shared_author_id'] and op['year']<=earliest['year'] and (
                    degree is None or (op['year']>=degree and not (same_phd and op['year']==degree))
                ):
                    row['operational_year']=op['year']
    dates.append(row)

with np.load(space_path,allow_pickle=False) as z:
    pos={w:i for i,w in enumerate(z['ids'].tolist())}
    all_v=z['vectors']
    assert z['dims']==128 and z['channel']=='refs'
papers=defaultdict(list)
for row in records:
    uid,w,yr=row
    if w in pos and np.linalg.norm(all_v[pos[w]])>0:
        papers[uid].append((w,yr,pos[w]))

def qhat(inds):
    vec=all_v[inds].astype(np.float64)
    n=len(inds)
    ss=(vec*vec).sum()
    v=vec.sum(axis=0)
    q=2*(n*ss-v@v)/(n*(n-1))
    return float(max(0,q))

def window_stats(uid,anchor,horizon,skip_event_year=False):
    result=defaultdict(list)
    for w,y,i in papers[uid]:
        e=y-anchor
        if -horizon<=e<horizon and (not skip_event_year or e!=0):
            result[e//2].append(i)
    return {j:qhat(ind) for j,ind in result.items() if len(ind)>=2}

def paired(date_key, horizon, common=True, skip_event_year=False):
    output=[]
    for d in dates:
        if d.get(date_key) is None or d['shared_author_id']:
            continue
        a,b=(window_stats(d['uid'],d[key],horizon,skip_event_year) for key in ('official_year',date_key))
        if common:
            a={j:q for j,q in a.items() if j in b}
            b={j:q for j,q in b.items() if j in a}
        def sides(w):
            return ([v for i,v in w.items() if i<0],[v for i,v in w.items() if i>=0])
        sides_a,sides_b=sides(a),sides(b)
        if not all((*sides_a,*sides_b)):
            continue
        vals=[float(np.mean(x)) for x in (*sides_a,*sides_b)]
        output.append({**d,'a_pre':vals[0],'a_post':vals[1],'b_pre':vals[2],'b_post':vals[3],
                       'official_delta':vals[1]-vals[0],'paper_delta':vals[3]-vals[2],
                       'common_pre_bins':len(sides_a[0]),'common_post_bins':len(sides_a[1])})
    return output

def lag(x, key='paper_year'):
    ok=[d[key]-d['official_year'] for d in x if d.get(key) is not None and not d['shared_author_id']]
    if not ok:
        return {'n':len(x),'linked':0}
    v=np.asarray(ok)
    return {'n':len(x),'linked':len(ok),'median':float(np.median(v)),'mean':float(v.mean()),
            'mae':float(np.abs(v).mean()),'min':int(v.min()),'max':int(v.max()),
            'lt':int((v<0).sum()),'eq':int((v==0).sum()),'gt':int((v>0).sum()),
            'abs_le1':int((np.abs(v)<=1).sum()),'abs_le2':int((np.abs(v)<=2).sum())}

def inference(sample,date_key):
    if not sample:
        return {'n':0}
    a=np.asarray([r['official_delta'] for r in sample])
    b=np.asarray([r['paper_delta'] for r in sample])
    v=np.stack((a,b,b-a),axis=1)
    field=np.array([r['field'] for r in sample])
    idx=[np.flatnonzero(field==f) for f in sorted(set(field))]
    ci=None
    if len(sample)>=5:
        rng=np.random.default_rng(7212)
        resample=np.concatenate([rng.choice(i,(10000,len(i)),replace=True) for i in idx],axis=1)
        ci=np.quantile(v[resample].mean(axis=1),[.025,.975],axis=0).T.tolist()
    return {'n':len(sample),'shifted_n':sum(r['official_year']!=r[date_key] for r in sample),
            'mean_pre':np.mean([[r['a_pre'],r['b_pre']] for r in sample],axis=0).tolist(),
            'mean_post':np.mean([[r['a_post'],r['b_post']] for r in sample],axis=0).tolist(),
            'delta':v.mean(axis=0).tolist(),'ci':ci,
            'sign_flips':int((a*b< -1e-12).sum()),
            'non_identical':int((np.abs(v[:,2])>1e-8).sum())}

primary=paired('paper_year',10)
settings={
    'main':(primary,'paper_year'),
    'independent_bins':(paired('paper_year',10,False),'paper_year'),
    'near':(paired('paper_year',4),'paper_year'),
    'skip_zero':(paired('paper_year',10,skip_event_year=True),'paper_year'),
    'same_primary_shifted':([r for r in primary if r['paper_year']!=r['official_year']],'paper_year'),
    'roster_proxy':(paired('operational_year',10),'operational_year'),
    'spring':([r for r in primary if r['spring']],'paper_year'),
}
fields=('mathematics','physics','chemistry','biology')
summary={
 'lag':{**{f:lag([d for d in dates if d['field']==f]) for f in fields},'all':lag(dates)},
 'roster_lag':{**{f:lag([d for d in dates if d['field']==f],'operational_year') for f in fields},'all':lag(dates,'operational_year')},
 'unavailable':dict(Counter(d['availability'] for d in dates)),
 'shared':sum(d['shared_author_id'] for d in dates),
 'roster_same_inst':sum(bool(d.get('roster_matches_official')) for d in dates),
 'eligible_according_to_prior':sum(r['eligible'] for r in official.values()),
 'has_coordinate_papers':len(papers),
 'papers_with_coordinates':sum(map(len,papers.values())),
 'reference_papers':len(records),
}
for name,(sample,datekey) in settings.items():
    summary[name]={**{f:inference([d for d in sample if d['field']==f],datekey) for f in fields},'all':inference(sample,datekey)}
triple=[]
for d in dates:
    if d['shared_author_id'] or d['paper_year'] is None or d.get('operational_year') is None:
        continue
    windows=[window_stats(d['uid'],d[k],10) for k in ('official_year','paper_year','operational_year')]
    indices=set.intersection(*(set(w) for w in windows))
    pre,post=sorted(j for j in indices if j<0),sorted(j for j in indices if j>=0)
    if not pre or not post:
        continue
    row=d.copy()
    for prefix,w in zip(('a','b','c'),windows):
        row[prefix+'_pre']=float(np.mean([w[j] for j in pre]))
        row[prefix+'_post']=float(np.mean([w[j] for j in post]))
    row['official_delta']=row['a_post']-row['a_pre']
    row['paper_delta']=row['b_post']-row['b_pre']
    row['operational_delta']=row['c_post']-row['c_pre']
    triple.append(row)

def roster_comparison(sample):
    mapped=[{**d,'b_pre':d['c_pre'],'b_post':d['c_post'],'paper_delta':d['operational_delta']} for d in sample]
    return inference(mapped,'operational_year')

summary['triple'] = {
    'official_institution':inference(triple,'paper_year'),
    'roster_institution':roster_comparison(triple),
    'roster_matches_official':sum(d['roster_matches_official'] for d in triple),
    'roster_differs':roster_comparison([d for d in triple if not d['roster_matches_official']]),
}
summary['roster_by_institution_match'] = {
    'same':inference([d for d in settings['roster_proxy'][0] if d['roster_matches_official']],'operational_year'),
    'different':inference([d for d in settings['roster_proxy'][0] if not d['roster_matches_official']],'operational_year'),
}
summary['primary_calendar'] = {
    'date_shift_gt2':sum(abs(d['paper_year']-d['official_year'])>2 for d in primary),
    'date_shift_min':min(d['paper_year']-d['official_year'] for d in primary),
    'date_shift_max':max(d['paper_year']-d['official_year'] for d in primary),
    'total_common_pre_bins':sum(d['common_pre_bins'] for d in primary),
    'total_common_post_bins':sum(d['common_post_bins'] for d in primary),
}
assert all(abs(d['official_delta']-d['paper_delta'])<1e-10 for d in primary if d['official_year']==d['paper_year'])
test=next(x for x in papers.values() if len(x)>2)[:3]
j=[x[2] for x in test]
v=all_v[j].astype(np.float64)
assert abs(qhat(j)-np.mean([np.sum((v[a]-v[b])**2) for a in range(3) for b in range(3) if a!=b]))<1e-12
(out/'dates.json').write_text(json.dumps(dates,ensure_ascii=False,indent=2)+'\n')
(out/'pairs.json').write_text(json.dumps(primary,ensure_ascii=False,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
