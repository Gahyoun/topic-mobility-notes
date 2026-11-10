"""Copy the private portfolio database and restrict its coordinate table to the audited GCC."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def event_presence(database):
    """Count any observed pre/post publications without the four-window requirements."""
    sql = """
    WITH faculty AS (
      SELECT o.uid,o.field,
             count(p.work) FILTER (WHERE p.year<o.appointment_year) AS pre,
             count(p.work) FILTER (WHERE p.year>o.appointment_year) AS post_1,
             count(p.work) FILTER (WHERE p.year>=o.appointment_year+3) AS post_3,
             count(c.work) FILTER (WHERE p.year<o.appointment_year) AS gcc_pre,
             count(c.work) FILTER (WHERE p.year>o.appointment_year) AS gcc_post_1,
             count(c.work) FILTER (WHERE p.year>=o.appointment_year+3) AS gcc_post_3
      FROM faculty_observation o
      LEFT JOIN publication_metadata p USING(uid)
      LEFT JOIN dag_native_coordinates c ON c.work=p.work
      WHERE o.n_supported_refs>0 AND o.appointment_year IS NOT NULL
      GROUP BY o.uid,o.field
    )
    SELECT coalesce(field,'all') AS field,count(*) AS dated,
           sum(pre>=1 AND post_1>=1) AS linked_plus1,
           sum(pre>=1 AND post_3>=1) AS linked_plus3,
           sum(gcc_pre>=1 AND gcc_post_1>=1) AS positioned_plus1,
           sum(gcc_pre>=1 AND gcc_post_3>=1) AS positioned_plus3,
           sum(gcc_pre>=2 AND gcc_post_3>=2) AS positioned_two_each
    FROM (SELECT * FROM faculty UNION ALL SELECT uid,NULL,pre,post_1,post_3,
          gcc_pre,gcc_post_1,gcc_post_3 FROM faculty)
    GROUP BY field
    """
    ro=database.resolve().as_uri()+'?mode=ro&immutable=1'
    with sqlite3.connect(ro,uri=True) as db:
        db.row_factory=sqlite3.Row
        return {r['field']:{k:r[k] for k in r.keys() if k!='field'} for r in db.execute(sql)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('portfolio','membership','output'):
        p.add_argument(k,type=Path)
    p.add_argument("--event-coverage",type=Path)
    a=p.parse_args()
    if a.output.exists():
        p.error('Choose a new private output database.')
    with a.membership.open(newline='') as f:
        rows=list(csv.DictReader(f))
    keep=[(r['work'],) for r in rows if r['earlier_year_with_external_sinks']=='True']
    ro=a.portfolio.resolve().as_uri()+'?mode=ro&immutable=1'
    with sqlite3.connect(ro,uri=True) as original, sqlite3.connect(a.output) as db:
        original.backup(db)
        db.execute('CREATE TEMP TABLE gcc(work TEXT PRIMARY KEY) WITHOUT ROWID')
        db.executemany('INSERT INTO gcc VALUES (?)',keep)
        removed=db.execute('DELETE FROM dag_native_coordinates WHERE work NOT IN (SELECT work FROM gcc)').rowcount
        remain=db.execute('SELECT count(*) FROM dag_native_coordinates').fetchone()[0]
        assert remain+removed==original.execute('SELECT count(*) FROM dag_native_coordinates').fetchone()[0]
        assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    if a.event_coverage:
        a.event_coverage.write_text(json.dumps(event_presence(a.output),indent=2)+'\n')
    print(f'GCC coordinates {remain:,}; outside the GCC {removed:,}.',flush=True)


if __name__=='__main__':
    main()
