from pathlib import Path
import sqlite3,json,sys
from collections import Counter,defaultdict
home=Path.home()
web=Path('Data/professor-career-dashboard/.private/mijin_publish_20261105/web.sqlite')
cohort=home/'work/topic-mobility-private/outputs/S14_full_roster_20261109/cohort-2025.sqlite'
def uri(p):return p.resolve().as_uri()+'?mode=ro&immutable=1'
with sqlite3.connect(uri(cohort),uri=True) as c:
 c.row_factory=sqlite3.Row
 c.execute('ATTACH DATABASE ? AS web',(uri(web),))
 rows=[dict(r) for r in c.execute("""
   WITH spring AS (SELECT DISTINCT professor_uid FROM web.semester_snapshots WHERE term='2026-spring')
   SELECT p.professor_uid uid, p.name, p.subject, p.latest_institution_raw school, p.phd_year, p.latest_rank rank,
          r.roster_year, r.first_ranked_roster_year, r.reconstructed_year, r.date_basis
   FROM roster_coverage r JOIN spring s USING(professor_uid) JOIN web.professors p USING(professor_uid)
   WHERE r.n_supported_refs>0 AND r.evidence_first_year IS NULL
   ORDER BY p.subject,p.latest_institution_raw,p.name
 """)]
assert len(rows)==791
path=Path(sys.argv[1])
(path/'queue.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(Counter(r['subject'] for r in rows))
print(Counter(r['school'] for r in rows).most_common(26))
print(Counter((r['school'],r['subject']) for r in rows).most_common(38))
print(Counter(r['date_basis'] for r in rows))
