import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

work = Path.cwd()
home = Path.home()
private = home / 'work/topic-mobility-private/outputs/S14_full_roster_20261109'
web = work / 'Data/professor-career-dashboard/.private/mijin_publish_20261105/web.sqlite'

def readonly(path):
    return path.resolve().as_uri() + '?mode=ro&immutable=1'

root = Path(__file__).parent
queue = {row['uid']:row for row in json.loads((root/'queue.json').read_text())}
with (root/'decisions.tsv').open() as stream:
    decision_rows = list(csv.DictReader(stream,delimiter='\t'))
decisions = {row['uid']:row for row in decision_rows}
assert len(decisions)==len(decision_rows)
assert set(decisions)<=set(queue)
verified = {uid:row for uid,row in decisions.items() if row['status']=='confirmed'}
assert all(verified[uid]['url'].startswith('https://') and verified[uid]['basis'] for uid in verified)
for uid,row in verified.items():
    assert 1950<=int(row['year'])<=2026
    ranked=queue[uid]['first_ranked_roster_year']
    assert ranked is None or int(row['year'])<=ranked, uid

with sqlite3.connect(readonly(private / 'cohort-2025.sqlite'), uri=True) as db:
    db.row_factory = sqlite3.Row
    db.execute('ATTACH DATABASE ? AS web', (readonly(web),))
    db.execute('CREATE TEMP TABLE verified (uid TEXT PRIMARY KEY, year INTEGER NOT NULL)')
    db.executemany('INSERT INTO verified VALUES (?,?)', ((uid,int(row['year'])) for uid,row in verified.items()))
    rows = db.execute("""
        WITH spring AS (
            SELECT DISTINCT professor_uid, institution_raw
            FROM web.semester_snapshots WHERE term='2026-spring'
        ), schools AS (
            SELECT professor_uid, GROUP_CONCAT(institution_raw, ' · ') school
            FROM (SELECT * FROM spring ORDER BY institution_raw)
            GROUP BY professor_uid
        ), updated AS (
            SELECT r.professor_uid, COALESCE(v.year,r.evidence_first_year) AS event_year,
                   v.uid IS NOT NULL AS verified, r.shared_author_id
            FROM roster_coverage r LEFT JOIN verified v ON v.uid=r.professor_uid
        ), bins AS (
            SELECT f.professor_uid, CAST((f.year-u.event_year+10)/2 AS INTEGER)*2-10 AS relative_year,
                   COUNT(*) AS papers
            FROM faculty_publication f JOIN updated u USING(professor_uid)
            JOIN reference_coverage c USING(work_id)
            WHERE u.shared_author_id=0 AND u.event_year IS NOT NULL
              AND f.year-u.event_year BETWEEN -10 AND 9
              AND f.review_only=0 AND c.state='cached_with_refs'
            GROUP BY f.professor_uid, relative_year
        ), windows AS (
            SELECT professor_uid, SUM(relative_year<0 AND papers>=2) AS pre,
                   SUM(relative_year>=0 AND papers>=2) AS post
            FROM bins GROUP BY professor_uid
        ), ref AS (
            SELECT professor_uid,
                   SUM(c.state='cached_no_refs') empty_records,
                   SUM(c.state NOT IN ('cached_no_refs','cached_with_refs')) unavailable
            FROM faculty_publication f JOIN reference_coverage c USING(work_id)
            WHERE f.review_only=0 GROUP BY professor_uid
        )
        SELECT p.professor_uid, p.name, COALESCE(s.school, p.latest_institution_raw) school,
               s.professor_uid IS NOT NULL spring, p.latest_term,
               r.subject, r.n_linked, r.n_supported, r.n_supported_refs, r.openalex_id_assigned,
               r.roster_year, u.event_year AS evidence_first_year,
               CASE WHEN u.verified THEN 'official_chronology' ELSE r.date_basis END AS date_basis,
               r.first_ranked_roster_year, r.reconstructed_year, r.shared_author_id,
               COALESCE(w.pre,0) pre, COALESCE(w.post,0) post,
               COALESCE(ref.empty_records,0) empty_records, COALESCE(ref.unavailable,0) unavailable
        FROM roster_coverage r JOIN web.professors p USING(professor_uid)
        JOIN updated u USING(professor_uid) LEFT JOIN schools s USING(professor_uid)
        LEFT JOIN windows w USING(professor_uid)
        LEFT JOIN ref USING(professor_uid)
        ORDER BY p.name, school, p.professor_uid
    """).fetchall()

def first_stop(r):
    if not r['n_supported']:
        if r['n_linked']:
            return 1, 'review'
        if not r['openalex_id_assigned']:
            return 1, 'no_id'
        return 1, 'no_paper'
    if not r['n_supported_refs']:
        return 2, 'no_refs'
    if r['evidence_first_year'] is None:
        return 3, r['date_basis']
    if r['shared_author_id']:
        return 4, 'shared_id'
    if not r['pre'] and not r['post']:
        return 4, 'both_windows'
    if not r['pre']:
        return 4, 'before_window'
    if not r['post']:
        return 4, 'after_window'
    return 5, 'included'

records = []
for r in rows:
    stage, reason = first_stop(r)
    records.append({
        'uid':r['professor_uid'], 'name':r['name'], 'school':r['school'], 'spring':bool(r['spring']),
        'lastTerm':r['latest_term'], 'field':r['subject'], 'stop':stage, 'reason':reason,
        'papers':r['n_supported'], 'refs':r['n_supported_refs'], 'roster':r['roster_year'],
        'event':r['evidence_first_year'], 'basis':r['date_basis'],
        'ranked':r['first_ranked_roster_year'], 'reconstructed':r['reconstructed_year'],
        'pre':None if not r['evidence_first_year'] or r['shared_author_id'] else r['pre'],
        'post':None if not r['evidence_first_year'] or r['shared_author_id'] else r['post'],
        'refEmpty':r['empty_records'], 'refUnavailable':r['unavailable'],
    })

def counts(data):
    return [sum(r['stop']>stage for r in data) for stage in range(5)]

spring = [r for r in records if r['spring']]
base_source = (private/'faculty-alluvial-2026-spring.html').read_text()
base = json.loads(re.search(r'<script type="application/json" id="records">(.*?)</script>',base_source,re.S)[1])
assert len(base)==len(records)==3937
for prior,now in zip(base,records):
    assert all(prior[k]==now[k] for k in ('name','school','spring','field'))
    if now['uid'] not in verified:
        assert prior=={k:v for k,v in now.items() if k!='uid'}
    else:
        assert prior['spring'] and prior['stop']==3 and now['stop']>=4
fields = ('mathematics','physics','chemistry','biology')
summary = {}
for field in fields:
    target = [uid for uid,row in queue.items() if row['subject']==field]
    recovered = [uid for uid in target if uid in verified]
    old = sum(r['spring'] and r['field']==field and r['stop']==5 for r in base)
    cur = sum(r['spring'] and r['field']==field and r['stop']==5 for r in records)
    reviewed = sum(uid in decisions for uid in target)
    summary[field] = {
        'target':len(target), 'reviewed':reviewed,
        'recovered':len(recovered), 'pending':len(target)-len(recovered),
        'reviewed_pending':reviewed-len(recovered), 'not_reviewed':len(target)-reviewed,
        'comparison_before':old, 'comparison_after':cur,
    }
assert sum(d['target'] for d in summary.values())==791
assert sum(d['recovered'] for d in summary.values())==len(verified)
assert counts(spring)[3]==2358+len(verified)
assert counts(records)[3]==2475+len(verified)
(root/'summary.json').write_text(json.dumps({'field':summary,'spring_counts':counts(spring),'all_counts':counts(records)},ensure_ascii=False,indent=2)+'\n')
(root/'updated-records.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')))
print(json.dumps(summary,ensure_ascii=False))
