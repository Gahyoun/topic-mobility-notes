"""Compute paper-based event years as a separate overlay of the appointment audit."""
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

here = Path(__file__).resolve().parent
prior = Path.home() / 'work/topic-mobility-private/outputs/S16_spring_appointment_audit_20261109'
s14 = prior.parent / 'S14_full_roster_20261109'
source = Path.cwd() / 'Data/professor-career-dashboard/.private'
db_path = source / 'mijin_publish_20261105/web.sqlite'
refs_path = prior.parent.parent / 'refs/work_references.sqlite'
s16_counts = json.loads((prior / 'summary.json').read_text())

with (prior / 'decisions.tsv').open() as f:
    decisions = {r['uid']: r for r in csv.DictReader(f, delimiter='\t')}

# Official-source adjudications that recorded an identity or date conflict, or
# described a possible earlier regular faculty appointment at another institution.
protected = {
    'prof:physics:phy0320', 'prof:chemistry:chemi0864', 'prof:mathematics:math0618',
    'prof:chemistry:chemi0388', 'prof:mathematics:math0615', 'prof:biology:bio0372',
    'prof:biology:bio0950', 'prof:physics:phy0673', 'prof:physics:phy0125',
    'prof:mathematics:math0526', 'prof:mathematics:math0249', 'prof:biology:bio0097',
    'prof:physics:phy0687', 'prof:biology:bio1433', 'prof:chemistry:chemi0714',
    'prof:chemistry:chemi0847',
}
assert all(decisions[uid]['status'] == 'ambiguous' for uid in protected)

def ro(path):
    return path.resolve().as_uri() + '?mode=ro&immutable=1'

def wid(value):
    key = (value or '').strip().split('/')[-1].upper()
    return key if re.fullmatch(r'W\d+', key) else None

with sqlite3.connect(ro(s14 / 'cohort-2025.sqlite'), uri=True) as db:
    db.row_factory = sqlite3.Row
    db.execute('ATTACH DATABASE ? AS web', (ro(db_path),))
    db.execute('ATTACH DATABASE ? AS refs', (ro(refs_path),))
    aliases = {wid(a):wid(b) for a,b in db.execute('SELECT requested_id,canonical_id FROM refs.work_alias')
               if wid(a) and wid(b) and wid(a)!=wid(b)}
    def canonical(value):
        key = wid(value)
        seen = set()
        while key in aliases:
            assert key not in seen
            seen.add(key)
            key = aliases[key]
        return key
    db.create_function('canonical_work', 1, canonical, deterministic=True)
    db.execute('CREATE TEMP TABLE audit_year(uid TEXT PRIMARY KEY, year INTEGER)')
    db.executemany('INSERT INTO audit_year VALUES (?,?)', ((uid,int(r['year']))
                    for uid,r in decisions.items() if r['status']=='confirmed'))
    db.executescript("""
        CREATE TEMP TABLE roster_institution AS
        WITH ordered AS (
          SELECT s.professor_uid, s.year*2+(s.term LIKE '%-fall') AS term_order,
                 s.year, s.institution_unit_id unit,
                 COALESCE(i.campus_cluster_id,s.institution_unit_id) campus,
                 MIN(CASE WHEN s.rank IN ('assistant','associate','full') THEN 0 ELSE 1 END)
                 OVER (PARTITION BY s.professor_uid) AS best_rank_type,
                 CASE WHEN s.rank IN ('assistant','associate','full') THEN 0 ELSE 1 END AS rank_type
          FROM web.semester_snapshots s
          JOIN web.institution_units i ON i.unit_id=s.institution_unit_id
        ), first_term AS (
          SELECT professor_uid, MIN(term_order) AS term_order FROM ordered
          WHERE rank_type=best_rank_type GROUP BY professor_uid
        )
        SELECT o.professor_uid, MIN(o.unit) AS unit, MIN(o.campus) AS campus,
               MIN(o.year) AS roster_cap, COUNT(DISTINCT o.campus) AS n_campuses,
               MIN(o.rank_type)=0 AS ranked
        FROM ordered o JOIN first_term t USING(professor_uid,term_order)
        WHERE rank_type=best_rank_type GROUP BY o.professor_uid;

        CREATE TEMP TABLE base AS
        SELECT r.*,p.name,p.latest_institution_raw,p.phd_year,i.unit,i.campus,i.roster_cap,i.n_campuses,i.ranked,
               COALESCE(a.year,r.evidence_first_year) AS documented_year,
               EXISTS(SELECT 1 FROM web.semester_snapshots s
                      WHERE s.professor_uid=r.professor_uid AND s.term='2026-spring') AS spring,
               EXISTS(SELECT 1 FROM web.education e JOIN web.institution_units u ON u.unit_id=e.institution_unit_id
                      WHERE e.professor_uid=r.professor_uid AND e.degree_level='phd'
                        AND (u.unit_id=i.unit OR u.campus_cluster_id=i.campus)) AS same_phd_campus
        FROM roster_coverage r JOIN web.professors p USING(professor_uid)
        LEFT JOIN roster_institution i USING(professor_uid)
        LEFT JOIN audit_year a ON a.uid=r.professor_uid;
        CREATE UNIQUE INDEX temp.base_uid ON base(professor_uid);

        CREATE TEMP TABLE affiliation_candidate AS
        SELECT b.professor_uid,f.work_id,f.year,b.unit,
               MIN(CASE WHEN u.institution_unit_id=b.unit THEN 0 ELSE 1 END) AS matched_via_campus
        FROM base b JOIN web.raw_affiliation_units u USING(professor_uid)
        JOIN faculty_publication f ON f.professor_uid=b.professor_uid AND f.work_id=canonical_work(u.work_id)
        WHERE b.documented_year IS NULL AND b.n_supported_refs>0 AND b.n_campuses=1
          AND b.shared_author_id=0 AND f.review_only=0 AND u.identity_decision='keep'
          AND (u.institution_unit_id=b.unit OR u.campus_cluster_id=b.campus)
        GROUP BY b.professor_uid,f.work_id,f.year,b.unit;
        CREATE UNIQUE INDEX temp.match_uid_work ON affiliation_candidate(professor_uid,work_id);
    """)
    source_names = dict(db.execute('SELECT unit_id,display_name FROM web.institution_units'))
    historical = {r['professor_uid']: (r['first_year'], r['source_url'], r['source_note'])
                  for r in db.execute('SELECT * FROM official_date')}
    base = [dict(r) for r in db.execute('SELECT * FROM base')]
    candidates = {r['professor_uid']:dict(r) for r in db.execute("""
        SELECT * FROM (
           SELECT a.*, ROW_NUMBER() OVER (PARTITION BY professor_uid
                  ORDER BY year,matched_via_campus,work_id) AS earliest
           FROM affiliation_candidate a
        ) WHERE earliest=1
    """)}
    overlay = {}
    holds = {}
    for r in base:
        uid = r['professor_uid']
        if r['documented_year'] is not None or not r['n_supported_refs']:
            continue
        first = candidates.get(uid)
        if r['shared_author_id']:
            holds[uid] = 'proxy_shared_id'
        elif uid in protected or r['conflicting_years']:
            holds[uid] = 'proxy_conflict'
        elif r['n_campuses'] != 1:
            holds[uid] = 'proxy_unclear_institution'
        elif first is None:
            holds[uid] = 'proxy_no_affiliated_paper'
        elif r['phd_year'] is not None and (
            first['year'] < r['phd_year'] or (r['same_phd_campus'] and first['year'] == r['phd_year'])
        ):
            holds[uid] = 'proxy_during_training'
        elif first['year'] > r['roster_cap']:
            holds[uid] = 'proxy_after_roster'
        else:
            assert 1950 <= first['year'] <= 2025
            overlay[uid] = {k:first[k] for k in ('work_id','year','unit','matched_via_campus')}
            overlay[uid].update(ranked=bool(r['ranked']), doctorate_year_available=r['phd_year'] is not None,
                                same_phd_campus=bool(r['same_phd_campus']), roster_cap=r['roster_cap'])
    db.execute('CREATE TEMP TABLE proxy_year(uid TEXT PRIMARY KEY, year INTEGER)')
    db.executemany('INSERT INTO proxy_year VALUES (?,?)', ((uid,r['year']) for uid,r in overlay.items()))
    windows = {r['professor_uid']:(r['pre'],r['post']) for r in db.execute("""
       WITH bin AS (
          SELECT p.uid AS professor_uid, CAST((f.year-p.year+10)/2 AS INTEGER)*2-10 AS rel
          FROM proxy_year p JOIN faculty_publication f ON f.professor_uid=p.uid
          JOIN reference_coverage c USING(work_id)
          WHERE f.review_only=0 AND c.state='cached_with_refs' AND f.year-p.year BETWEEN -10 AND 9
          GROUP BY p.uid, rel HAVING COUNT(*)>=2
       )
       SELECT professor_uid,SUM(rel<0) AS pre,SUM(rel>=0) AS post FROM bin GROUP BY professor_uid
    """)}
    documented = {r['professor_uid']:(r['pre'],r['post']) for r in db.execute("""
       WITH bin AS (
          SELECT b.professor_uid, CAST((f.year-b.documented_year+10)/2 AS INTEGER)*2-10 AS rel
          FROM base b JOIN faculty_publication f USING(professor_uid) JOIN reference_coverage c USING(work_id)
          WHERE b.shared_author_id=0 AND b.documented_year IS NOT NULL AND f.review_only=0
            AND c.state='cached_with_refs' AND f.year-b.documented_year BETWEEN -10 AND 9
          GROUP BY b.professor_uid, rel HAVING COUNT(*)>=2
       )
       SELECT professor_uid,SUM(rel<0) AS pre,SUM(rel>=0) AS post FROM bin GROUP BY professor_uid
    """)}

def stage(r, proxy):
    uid = r['professor_uid']
    if not r['n_supported']:
        return (1,'review') if r['n_linked'] else (1,'no_paper' if r['openalex_id_assigned'] else 'no_id')
    if not r['n_supported_refs']:
        return 2,'no_refs'
    if r['documented_year'] is None and not (proxy and uid in overlay):
        status=decisions.get(uid,{}).get('status')
        old_reason={'current_only':'official_current_only','ambiguous':'official_insufficient'}.get(status,r['date_basis'])
        return 3,holds[uid] if proxy else old_reason
    if r['shared_author_id']:
        return 4,'shared_id'
    pre,post=(windows if uid in overlay and proxy else documented).get(uid,(0,0))
    if pre and post:
        return 5,'included'
    return 4,'after_window' if pre else 'before_window' if post else 'both_windows'

fields = ('mathematics','physics','chemistry','biology')
totals = {}
for name in ('spring','all'):
    people = [r for r in base if name=='all' or r['spring']]
    records = {}
    for field in fields:
        members = [r for r in people if r['subject']==field]
        assert members
        old = [stage(r,False)[0] for r in members]
        new = [stage(r,True) for r in members]
        assert all(before<=after[0] for before,after in zip(old,new))
        records[field] = {
            'baseline':[sum(s>i for s in old) for i in range(5)],
            'counts':[sum(s>i for s,_ in new) for i in range(5)],
            'holds':[dict(Counter(reason for s,reason in new if s==i)) for i in range(1,5)],
            'proxy_holds_detailed': dict(Counter(holds[r['professor_uid']] for r in members if r['professor_uid'] in holds)),
        }
    assert [sum(r['baseline'][i] for r in records.values()) for i in range(5)] == s16_counts[name+'_counts']
    totals[name] = records
assert len(base)==3937
evidence = []
for r in base:
    uid = r['professor_uid']
    record = {'uid':uid,'name':r['name'],'field':r['subject'],'spring':int(r['spring']),
              'latest_institution':r['latest_institution_raw']}
    if uid in historical:
        year,url,note = historical[uid]
        institution = {
            'Andreas Heinrich':'Ewha Womans University', 'Fabio Donati':'Ewha Womans University',
            '김기용':'University of Maryland', '김창영':'연세대학교',
            '전헌수':'서울대학교', '노도영':'GIST',
        }[r['name']]
        record.update(kind='official',year=year,institution=institution,source_url=url,note=note,
                      work_id=None,unit_id=None,roster_year=None,ranked=None,eligible=int(stage(r,False)[0]==5))
    elif decisions.get(uid,{}).get('status')=='confirmed':
        dec=decisions[uid]
        record.update(kind='official',year=int(dec['year']),institution=dec['institution'],source_url=dec['url'],note=dec['basis'],
                      work_id=None,unit_id=None,roster_year=None,ranked=None,eligible=int(stage(r,False)[0]==5))
    elif uid in overlay:
        anchor=overlay[uid]
        record.update(kind='first_affiliated_paper',year=anchor['year'],institution=source_names[anchor['unit']],
                      source_url='https://openalex.org/'+anchor['work_id'],
                      note='같은 캠퍼스의 정규화된 기관 단위' if anchor['matched_via_campus'] else '정규화된 동일 기관 단위',
                      work_id=anchor['work_id'],unit_id=anchor['unit'],roster_year=anchor['roster_cap'],
                      ranked=int(anchor['ranked']),eligible=int(stage(r,True)[0]==5))
    else:
        continue
    evidence.append(record)
assert len(evidence)==len(overlay)+len(historical)+sum(r['status']=='confirmed' for r in decisions.values())
assert len({r['uid'] for r in evidence})==len(evidence)
(here/'evidence-export.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')

(here/'summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2)+'\n')
(here/'proxy-appointments.json').write_text(json.dumps(overlay,ensure_ascii=False,indent=2)+'\n')
print('Proxies:',len(overlay),'ranked anchor:',sum(r['ranked'] for r in overlay.values()),
      'campus matches:',sum(r['matched_via_campus'] for r in overlay.values()),
      'no doctorate year:',sum(not r['doctorate_year_available'] for r in overlay.values()))
for scope, data in totals.items():
    print(scope)
    for f,c in data.items():
        print(f,c['baseline'],c['counts'],c['proxy_holds_detailed'])
