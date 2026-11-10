"""Rebuild the 2025 roster frame and outgoing-reference coverage from read-only sources."""

import gzip
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

CUTOFF = 2025
PRIVATE = Path('Data/professor-career-dashboard/.private')
SOURCE = PRIVATE / 'mijin_publish_20261105'
OLD_ROOT = Path.home() / 'work/topic-mobility-private'
OLD_ARCHIVE = OLD_ROOT / 'outputs/S13_external_citations_20261108/archive_2025-12-31/openalex-topic-mobility-through-2025-12-31.sqlite'

# Earliest documented regular university faculty position in the published chronology.
# Department, doctoral year and source roster spelling disambiguate the record.
OFFICIAL = json.loads(Path(os.environ["TOPIC_MOBILITY_OFFICIAL_DATES"]).read_text())

def ro_uri(path):
    return path.resolve().as_uri() + '?mode=ro&immutable=1'

def wid(value):
    short = (value or '').strip().split('/')[-1].upper()
    return short if re.fullmatch(r'W\d+', short) else None

def year_as_of(work):
    return work.get('publication_year') and work['publication_year'] <= CUTOFF and (
        not work.get('publication_date') or work['publication_date'] <= '2025-12-31')

def make_database(db):
    with sqlite3.connect(db.resolve().as_uri(), uri=True) as con:
        for name, path in {
            'latest': SOURCE / 'web.sqlite',
            'koad': SOURCE / 'identity_decisions.sqlite',
            'refs': OLD_ROOT / 'refs/work_references.sqlite',
            'old': OLD_ARCHIVE,
        }.items():
            con.execute(f'ATTACH DATABASE ? AS {name}', (ro_uri(path),))
        con.executescript("""
        CREATE TABLE provenance(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE official_date(
            professor_uid TEXT PRIMARY KEY, first_year INTEGER NOT NULL,
            source_url TEXT NOT NULL, source_note TEXT NOT NULL,
            FOREIGN KEY(professor_uid) REFERENCES roster(professor_uid)
        );
        CREATE TABLE roster(
            professor_uid TEXT PRIMARY KEY, subject TEXT NOT NULL, roster_year INTEGER,
            first_roster_year INTEGER, first_roster_term TEXT, first_ranked_roster_year INTEGER, reconstructed_year INTEGER,
            reconstructed_confidence TEXT, reconstructed_status TEXT,
            conflicting_years INTEGER NOT NULL, shared_author_id INTEGER NOT NULL,
            openalex_id_assigned INTEGER NOT NULL
        );
        INSERT INTO roster
        WITH semesters AS (
          SELECT professor_uid, MIN(year) first_roster_year, MIN(term) first_roster_term,
                 MIN(CASE WHEN rank IN ('assistant','associate','full') THEN year END) first_ranked_roster_year
          FROM latest.semester_snapshots GROUP BY professor_uid
        ), dates AS (
          SELECT professor_uid,
            CASE WHEN COUNT(DISTINCT start_year)=1 THEN MIN(start_year) END AS year,
            CASE WHEN COUNT(DISTINCT start_year)=1 THEN
              CASE WHEN MAX(confidence='confirmed') THEN 'confirmed'
                   WHEN MAX(confidence='high') THEN 'high' ELSE 'medium' END END AS confidence,
            CASE WHEN COUNT(DISTINCT evidence_status)=1 THEN MIN(evidence_status) END AS status,
            COUNT(DISTINCT start_year)>1 AS conflict
          FROM latest.career_positions_v2
          WHERE position_type='faculty' AND position_no=1 AND start_year IS NOT NULL
          GROUP BY professor_uid
        ), shared AS (
          SELECT openalex_id FROM latest.professors
          WHERE openalex_id IS NOT NULL GROUP BY openalex_id HAVING COUNT(*)>1
        )
        SELECT p.professor_uid, p.subject, p.appointment_year, s.first_roster_year,
               s.first_roster_term, s.first_ranked_roster_year, d.year, d.confidence, d.status, COALESCE(d.conflict,0),
               h.openalex_id IS NOT NULL, p.openalex_id IS NOT NULL
        FROM latest.professors p
        LEFT JOIN semesters s USING(professor_uid)
        LEFT JOIN dates d USING(professor_uid)
        LEFT JOIN shared h USING(openalex_id);
        CREATE TABLE faculty_publication(
            professor_uid TEXT NOT NULL, work_id TEXT NOT NULL, year INTEGER NOT NULL,
            work_type TEXT NOT NULL, topic_id TEXT, review_only INTEGER NOT NULL,
            PRIMARY KEY(professor_uid, work_id)
        ) WITHOUT ROWID;
        """)
        for name, institution, phd, year, url, note in OFFICIAL:
            rows = con.execute("""
                SELECT professor_uid FROM latest.professors
                WHERE name=? AND latest_institution_raw=? AND phd_year=?
                      AND appointment_year IS NULL
                """, (name, institution, phd)).fetchall()
            if len(rows) != 1:
                raise ValueError(f'Official evidence target resolves to {len(rows)} rows')
            con.execute('INSERT INTO official_date VALUES(?,?,?,?)', (rows[0][0], year, url, note))
        aliases = {wid(a): wid(b) for a, b in con.execute('SELECT requested_id, canonical_id FROM refs.work_alias')
                   if wid(a) and wid(b) and wid(a) != wid(b)}
        def canonical(value):
            key = wid(value)
            visited = set()
            while key in aliases:
                if key in visited:
                    raise ValueError('Work alias cycle')
                visited.add(key)
                key = aliases[key]
            return key
        con.create_function('canonical', 1, canonical, deterministic=True)
        # The overlay is keyed to the unnormalized authorship evidence. Block its explicit
        # exclusions across aliases as well so an accepted copy cannot bypass the decision.
        con.executescript("""
        CREATE TEMP TABLE blocked AS
        SELECT DISTINCT professor_uid, canonical(work_id) work_id FROM koad.work_decision
        WHERE decision='namesake_paper' OR recommended_action='drop';
        CREATE INDEX temp.blocked_link ON blocked(professor_uid,work_id);
        CREATE TEMP TABLE reviewed_base AS
        SELECT w.professor_uid, canonical(w.work_id) work_id, w.publication_year year,
               w.work_type, w.topic_id,
               k.recommended_action='review'
                 AND NOT COALESCE(k.is_seed OR k.core_coauthor_connected
                                  OR (k.institution_connected AND k.department_compatible=1),0)
                 AS unanchored_review
        FROM latest.work_authorship_evidence w
        LEFT JOIN koad.work_decision k USING(professor_uid,work_id)
        WHERE w.identity_decision='keep' AND w.publication_year<=2025
          AND (w.publication_date IS NULL OR w.publication_date='' OR w.publication_date<='2025-12-31')
          AND canonical(w.work_id) IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM blocked b WHERE b.professor_uid=w.professor_uid AND b.work_id=canonical(w.work_id)
          );
        INSERT INTO faculty_publication
        SELECT professor_uid, work_id, MIN(year), MIN(work_type), MAX(topic_id),
               MIN(COALESCE(unanchored_review,0))
        FROM reviewed_base
        WHERE work_type IN ('article','review','letter','editorial')
        GROUP BY professor_uid,work_id
        HAVING COUNT(DISTINCT year)=1;
        CREATE INDEX publication_work ON faculty_publication(work_id);
        CREATE TABLE outgoing_reference(
            work_id TEXT NOT NULL, reference_id TEXT NOT NULL,
            PRIMARY KEY(work_id,reference_id)
        ) WITHOUT ROWID;
        CREATE TABLE reference_coverage(
            work_id TEXT PRIMARY KEY, state TEXT NOT NULL, source TEXT, n_references INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO reference_coverage(work_id,state)
        SELECT DISTINCT work_id,'not_in_local_cache' FROM faculty_publication;
        """)
        # A full cached OpenAlex record takes precedence over an empty or missing older record.
        # A nonempty list from the original cache remains available when the newer cache is empty.
        for file in sorted(PRIVATE.glob('kri_20261105/*/fullworks.jsonl.gz')) + [SOURCE / 'fullworks.jsonl.gz']:
            with gzip.open(file, 'rt') as handle:
                for line in handle:
                    work = json.loads(line)['work']
                    refs = work.get('referenced_works')
                    work_id = canonical(work.get('id'))
                    if refs is None or not work_id or not year_as_of(work):
                        continue
                    if not con.execute('SELECT 1 FROM reference_coverage WHERE work_id=?', (work_id,)).fetchone():
                        continue
                    if refs:
                        con.execute('DELETE FROM outgoing_reference WHERE work_id=?', (work_id,))
                        con.executemany('INSERT OR IGNORE INTO outgoing_reference VALUES (?,?)',
                            ((work_id, ref) for v in refs if (ref := canonical(v)) and ref != work_id))
                        con.execute("UPDATE reference_coverage SET state='cached_with_refs',source='2026-11-05 full works' WHERE work_id=?", (work_id,))
                    else:
                        con.execute("UPDATE reference_coverage SET state='cached_no_refs',source='2026-11-05 full works' WHERE work_id=? AND state<>'cached_with_refs'", (work_id,))
        con.executescript("""
        INSERT OR IGNORE INTO outgoing_reference
        SELECT c.work_id,canonical(r.referenced_work_id)
        FROM reference_coverage c JOIN refs.work_references r USING(work_id)
        WHERE c.state<>'cached_with_refs'
          AND canonical(r.referenced_work_id) IS NOT NULL
          AND canonical(r.referenced_work_id)<>c.work_id;
        UPDATE reference_coverage AS c
        SET n_references=(SELECT COUNT(*) FROM outgoing_reference r WHERE r.work_id=c.work_id);
        UPDATE reference_coverage AS c
        SET state=CASE WHEN c.n_references>0 THEN 'cached_with_refs'
                       WHEN c.state='cached_no_refs' THEN c.state
                       WHEN s.status='cached_with_refs' THEN 'unresolved_reference_edges'
                       ELSE COALESCE(s.status,c.state) END,
            source=COALESCE(c.source,'2026-11-04 reference cache')
        FROM refs.work_ref_status s WHERE s.work_id=c.work_id;
        UPDATE reference_coverage SET state='unresolved_reference_edges'
        WHERE state='cached_with_refs' AND n_references=0;
        CREATE TABLE roster_coverage AS
        WITH per_faculty AS (
            SELECT f.professor_uid,
                   COUNT(*) AS n_linked,
                   SUM(f.review_only=0) AS n_supported,
                   SUM(c.state='cached_with_refs') AS n_refs,
                   SUM(c.state='cached_with_refs' AND f.review_only=0) AS n_supported_refs,
                   SUM(f.topic_id IS NOT NULL) AS n_with_topic
            FROM faculty_publication f JOIN reference_coverage c USING(work_id) GROUP BY f.professor_uid
        )
        SELECT r.*, o.first_year AS official_first_year,
               CASE WHEN o.first_year IS NOT NULL THEN o.first_year
                    WHEN r.reconstructed_confidence IN ('high','confirmed') AND
                       (r.first_ranked_roster_year IS NULL OR r.reconstructed_year<=r.first_ranked_roster_year)
                    THEN r.reconstructed_year END AS evidence_first_year,
               CASE WHEN o.first_year IS NOT NULL THEN 'official_chronology'
                    WHEN r.reconstructed_confidence IN ('high','confirmed') AND r.reconstructed_year>r.first_ranked_roster_year
                    THEN 'reconstruction_after_observed_faculty_rank'
                    WHEN r.reconstructed_confidence IN ('high','confirmed') THEN 'roster_and_affiliation_reconstruction'
                    WHEN r.conflicting_years=1 THEN 'conflicting_reconstruction'
                    WHEN r.roster_year IS NOT NULL THEN 'roster_year_only_or_uncertain'
                    ELSE 'first_date_unavailable' END AS date_basis,
               COALESCE(p.n_linked,0) AS n_linked,
               COALESCE(p.n_supported,0) AS n_supported,
               COALESCE(p.n_refs,0) AS n_refs,
               COALESCE(p.n_supported_refs,0) AS n_supported_refs,
               COALESCE(p.n_with_topic,0) AS n_with_topic
        FROM roster r LEFT JOIN official_date o USING(professor_uid)
        LEFT JOIN per_faculty p USING(professor_uid);
        CREATE UNIQUE INDEX roster_coverage_uid ON roster_coverage(professor_uid);
        CREATE TABLE recovery_queue AS
        SELECT professor_uid, subject, 'no_supported_journal_link' AS reason
        FROM roster_coverage WHERE n_supported=0
        UNION ALL SELECT professor_uid, subject, 'first_year_missing_or_uncertain'
        FROM roster_coverage WHERE evidence_first_year IS NULL
        UNION ALL SELECT professor_uid, subject, 'shared_openalex_author_id'
        FROM roster_coverage WHERE shared_author_id=1
        UNION ALL SELECT professor_uid, subject, 'journal_links_but_no_usable_references'
        FROM roster_coverage WHERE n_supported>0 AND n_supported_refs=0;
        CREATE VIEW cohort_windows AS
        SELECT y.professor_uid, y.basis, y.event_year,
               CAST((f.year-y.event_year+10)/2 AS INTEGER)*2-10 AS event_window_start,
               COUNT(*) AS n_papers
        FROM (
          SELECT professor_uid, 'documented_or_reconstructed' AS basis, evidence_first_year AS event_year
          FROM roster_coverage WHERE evidence_first_year IS NOT NULL AND shared_author_id=0
          UNION ALL
          SELECT professor_uid, 'roster_year_sensitivity', COALESCE(official_first_year,roster_year)
          FROM roster_coverage WHERE (official_first_year IS NOT NULL OR roster_year IS NOT NULL) AND shared_author_id=0
        ) y
        JOIN faculty_publication f USING(professor_uid)
        JOIN reference_coverage c USING(work_id)
        WHERE f.year-y.event_year BETWEEN -10 AND 9
          AND f.review_only=0 AND c.state='cached_with_refs'
        GROUP BY y.professor_uid,y.basis,y.event_year,event_window_start;
        """)
        con.executemany('INSERT INTO provenance VALUES (?,?)', (
            ('publication_cutoff','2025-12-31'),
            ('latest_roster','mijin_publish_20261105/web.sqlite'),
            ('authorship_overlay','mijin_publish_20261105/identity_decisions.sqlite'),
            ('incoming_citations_used_for_inclusion','false'),
            ('first_appointment_rule','Official chronology or a consistent high-confidence reconstruction. Withhold reconstructed years later than the first roster observation with an explicit faculty rank.'),
            ('shared_author_id_policy','Flag 90 rows and withhold from individual transition diagnostics pending identity reconciliation. Keep all roster rows.'),
            ('reference_records','2026-11-04 work_references.sqlite and 2026-11-05 OpenAlex full works caches. Not an archived OpenAlex snapshot downloaded in 2025.'),
            ('window','-10 to +9 years from year of appointment in ten non-overlapping two-year windows. Each eligible window has at least two linked papers with references.'),
        ))
        summary = {}
        def scalar(sql):
            return con.execute(sql).fetchone()[0]
        summary['raw_accepted_roster'] = scalar('SELECT COUNT(DISTINCT professor_uid) FROM latest.work_authorship_evidence WHERE identity_decision=\'keep\' AND publication_year<=2025')
        summary['old_archive_roster'] = scalar('SELECT COUNT(DISTINCT professor_uid) FROM old.faculty_publication')
        summary['roster'] = scalar('SELECT COUNT(*) FROM roster')
        summary['official_dates_added'] = scalar('SELECT COUNT(*) FROM official_date')
        summary['date_after_ranked_roster'] = scalar("SELECT COUNT(*) FROM roster_coverage WHERE date_basis='reconstruction_after_observed_faculty_rank'")
        summary['linked_in_old_and_new'] = scalar('SELECT COUNT(*) FROM roster_coverage WHERE n_supported>0 AND EXISTS (SELECT 1 FROM old.faculty_publication o WHERE o.professor_uid=roster_coverage.professor_uid)')
        summary['supported_new_to_archive'] = scalar('SELECT COUNT(*) FROM roster_coverage WHERE n_supported>0 AND NOT EXISTS (SELECT 1 FROM old.faculty_publication o WHERE o.professor_uid=roster_coverage.professor_uid)')
        summary['new_work_links'] = scalar("SELECT COUNT(*) FROM faculty_publication f WHERE f.review_only=0 AND NOT EXISTS (SELECT 1 FROM old.faculty_publication o WHERE o.professor_uid=f.professor_uid AND o.work_id=CAST(substr(f.work_id,2) AS INTEGER))")
        summary['totals'] = con.execute("""
            SELECT SUM(openalex_id_assigned),SUM(n_linked>0),SUM(n_supported>0),SUM(n_supported_refs>0),
                   SUM(roster_year IS NOT NULL OR official_first_year IS NOT NULL),
                   SUM(evidence_first_year IS NOT NULL),SUM(evidence_first_year IS NOT NULL AND n_supported_refs>0),
                   SUM(shared_author_id),SUM(conflicting_years),SUM(n_supported),SUM(n_supported_refs)
            FROM roster_coverage
        """).fetchone()
        summary['field_breakdown'] = con.execute("""
            SELECT subject,COUNT(*),SUM(n_linked>0),SUM(n_supported>0),SUM(n_supported_refs>0),
                   SUM(evidence_first_year IS NOT NULL),SUM(evidence_first_year IS NOT NULL AND n_supported_refs>0),
                   SUM(n_supported=0 AND evidence_first_year IS NULL),SUM(shared_author_id)
            FROM roster_coverage GROUP BY subject ORDER BY subject
        """).fetchall()
        summary['work_refs'] = con.execute('SELECT state,COUNT(*),SUM(n_references) FROM reference_coverage GROUP BY state ORDER BY state').fetchall()
        summary['transition'] = con.execute("""
            SELECT basis,SUM(pre>0 AND post>0),SUM(pre>=2 AND post>=2)
            FROM (
               SELECT basis,professor_uid,
                      SUM(event_window_start<0 AND n_papers>=2) pre,
                      SUM(event_window_start>=0 AND n_papers>=2) post
               FROM cohort_windows GROUP BY basis,professor_uid
            ) GROUP BY basis ORDER BY basis
        """).fetchall()
        assert summary['roster']==3937
        assert scalar('SELECT COUNT(*) FROM roster_coverage')==summary['roster']
        assert scalar('SELECT COUNT(*) FROM faculty_publication f LEFT JOIN roster r USING(professor_uid) WHERE r.professor_uid IS NULL')==0
        assert scalar('SELECT COUNT(*) FROM faculty_publication WHERE year>2025')==0
        assert scalar("SELECT COUNT(*) FROM outgoing_reference WHERE work_id=reference_id")==0
        assert scalar("SELECT COUNT(*) FROM reference_coverage WHERE (state='cached_with_refs')<>(n_references>0)")==0
        assert scalar('SELECT COUNT(*) FROM roster_coverage WHERE evidence_first_year>first_ranked_roster_year')==0
        assert scalar('PRAGMA integrity_check')=='ok'
        return summary

if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(mode=0o700)
    summary = make_database(out / 'cohort-2025.sqlite')
    result = json.dumps(summary, ensure_ascii=False, indent=2)
    (out / 'summary.json').write_text(result+'\n')
    print(result)
