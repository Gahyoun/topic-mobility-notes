"""Add traceable event-year evidence to the existing local career database."""
import hashlib
import json
import sqlite3
from pathlib import Path

here = Path(__file__).resolve().parent
target = Path.cwd() / 'DataBase/professor_careers_2023_2026.sqlite'
evidence = json.loads((here / 'evidence-export.json').read_text())
columns = ('uid','kind','year','institution','source_url','note','work_id','unit_id','roster_year','ranked','spring','eligible')
values = [tuple(r[k] for k in columns) for r in evidence]

def original_years(db):
    rows = db.execute('SELECT professor_uid, appointment_year FROM professors ORDER BY professor_uid').fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()

with sqlite3.connect(target.resolve().as_uri()+'?mode=rw', uri=True, timeout=15) as db:
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('BEGIN IMMEDIATE')
    before = original_years(db)
    people = {uid:(name,field) for uid,name,field in db.execute('SELECT professor_uid,name,subject FROM professors')}
    assert all(people.get(r['uid'])==(r['name'],r['field']) for r in evidence)
    db.execute("""
        CREATE TABLE IF NOT EXISTS faculty_appointment_evidence (
          professor_uid TEXT NOT NULL REFERENCES professors(professor_uid),
          evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('official','first_affiliated_paper')),
          event_year INTEGER NOT NULL CHECK(event_year BETWEEN 1950 AND 2026),
          institution_label TEXT NOT NULL,
          source_url TEXT NOT NULL,
          source_note TEXT NOT NULL,
          openalex_work_id TEXT,
          institution_unit_id TEXT,
          institution_first_roster_year INTEGER,
          roster_rank_recorded INTEGER CHECK(roster_rank_recorded IN (0,1)),
          in_spring_2026 INTEGER NOT NULL CHECK(in_spring_2026 IN (0,1)),
          can_compare_2year_windows INTEGER NOT NULL CHECK(can_compare_2year_windows IN (0,1)),
          PRIMARY KEY (professor_uid,evidence_kind),
          CHECK((evidence_kind='official' AND openalex_work_id IS NULL)
            OR (evidence_kind='first_affiliated_paper' AND openalex_work_id IS NOT NULL
                AND institution_unit_id IS NOT NULL AND institution_first_roster_year IS NOT NULL
                AND roster_rank_recorded IS NOT NULL AND event_year<=institution_first_roster_year))
        )
    """)
    db.executemany("""
        INSERT INTO faculty_appointment_evidence (
          professor_uid,evidence_kind,event_year,institution_label,source_url,source_note,
          openalex_work_id,institution_unit_id,institution_first_roster_year,roster_rank_recorded,
          in_spring_2026,can_compare_2year_windows
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(professor_uid,evidence_kind) DO UPDATE SET
          event_year=excluded.event_year,institution_label=excluded.institution_label,
          source_url=excluded.source_url,source_note=excluded.source_note,
          openalex_work_id=excluded.openalex_work_id,institution_unit_id=excluded.institution_unit_id,
          institution_first_roster_year=excluded.institution_first_roster_year,
          roster_rank_recorded=excluded.roster_rank_recorded,in_spring_2026=excluded.in_spring_2026,
          can_compare_2year_windows=excluded.can_compare_2year_windows
    """, values)
    db.execute("""
        CREATE VIEW IF NOT EXISTS v_faculty_appointment_evidence AS
        SELECT p.name,p.subject,p.appointment_year AS original_appointment_year,e.*
        FROM faculty_appointment_evidence e JOIN professors p USING(professor_uid)
    """)
    recorded = set(db.execute("""
        SELECT professor_uid,evidence_kind,event_year,institution_label,source_url,source_note,
               openalex_work_id,institution_unit_id,institution_first_roster_year,roster_rank_recorded,
               in_spring_2026,can_compare_2year_windows
        FROM faculty_appointment_evidence
    """))
    assert set(values)<=recorded and original_years(db)==before
    assert not db.execute('PRAGMA foreign_key_check(faculty_appointment_evidence)').fetchall()

with sqlite3.connect(target.resolve().as_uri()+'?mode=ro', uri=True) as db:
    actual = db.execute('SELECT evidence_kind,COUNT(*),SUM(in_spring_2026) FROM v_faculty_appointment_evidence GROUP BY 1 ORDER BY 1').fetchall()
    assert actual==[('first_affiliated_paper',483,363),('official',100,99)]
    assert original_years(db)==before and db.execute('PRAGMA quick_check').fetchone()==('ok',)
    print(actual)
