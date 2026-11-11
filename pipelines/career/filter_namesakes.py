"""Apply KOAD high-confidence drops and adjudicated professor–work corrections to S14."""
import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory


def read_only(path):
    return path.resolve().as_uri() + '?mode=ro'


def work_id(value):
    match = re.fullmatch(r'(?:https?://openalex\.org/)?(W[1-9][0-9]*)', value, re.IGNORECASE)
    if match is None:
        raise ValueError(f'Invalid OpenAlex work ID: {value!r}')
    return match[1].upper()


def canonicalizer(db):
    aliases = {work_id(a): work_id(b) for a, b in db.execute(
        'SELECT requested_id, canonical_id FROM work_alias WHERE requested_id<>canonical_id'
    )}
    def canonical(value):
        key = work_id(value)
        seen = set()
        while key in aliases and aliases[key] != key:
            if key in seen:
                raise ValueError(f'Work alias cycle at {key}')
            seen.add(key)
            key = aliases[key]
        return key
    return canonical


def candidates(decision_db, metadata_db, adjudications, canonical):
    confirmed = defaultdict(set)
    verified = set()
    with closing(sqlite3.connect(read_only(decision_db), uri=True)) as db:
        for uid, work in db.execute("""SELECT professor_uid,work_id FROM work_decision
              WHERE decision='namesake_paper' AND confidence='high' AND recommended_action='drop'"""):
            confirmed[uid, canonical(work)].add('koad_high_drop')
    with closing(sqlite3.connect(read_only(metadata_db), uri=True)) as db:
        for uid, work in db.execute("""SELECT professor_uid,json_extract(original_row_json,'$.work_id')
             FROM wrong_link_quarantine_audit WHERE table_name='work_authorship_evidence'"""):
            confirmed[uid, canonical(work)].add('documented_wrong_link')
    for row in json.loads(adjudications.read_text(encoding='utf-8')):
        key = row['professor_uid'], canonical(row['work_id'])
        if row['decision'] == 'keep':
            verified.add(key)
        elif row['decision'] == 'drop' and row['evidence'].strip():
            confirmed[key].add('verified_separate_identity')
        else:
            raise ValueError('Adjudications require keep or drop and a basis for each drop.')
    return {key: sorted(reasons) for key, reasons in confirmed.items() if key not in verified}


def filter_cohort(db, decisions):
    before = db.execute('SELECT COUNT(*) FROM faculty_publication').fetchone()[0]
    db.execute("""CREATE TABLE namesake_filter_audit (
        professor_uid TEXT,work_id TEXT,reasons_json TEXT NOT NULL,
        PRIMARY KEY(professor_uid,work_id)) WITHOUT ROWID""")
    matches = (row for row in db.execute(
        'SELECT professor_uid,work_id FROM faculty_publication WHERE review_only=0 ORDER BY 1,2'
    ) if row in decisions)
    db.executemany('INSERT INTO namesake_filter_audit VALUES (?,?,?)',
                   ((uid, work, json.dumps(decisions[uid, work])) for uid, work in matches))
    db.executescript("""
        DELETE FROM faculty_publication AS f WHERE EXISTS (
          SELECT 1 FROM namesake_filter_audit a WHERE a.professor_uid=f.professor_uid
            AND a.work_id=f.work_id);
        CREATE TEMP TABLE affected_coverage AS
        SELECT a.professor_uid, COUNT(f.work_id) AS linked,
               coalesce(sum(f.review_only=0),0) AS supported,
               coalesce(sum(c.state='cached_with_refs'),0) AS refs,
               coalesce(sum(c.state='cached_with_refs' AND f.review_only=0),0) AS supported_refs,
               coalesce(sum(f.topic_id IS NOT NULL),0) AS topics
        FROM (SELECT DISTINCT professor_uid FROM namesake_filter_audit) a
        LEFT JOIN faculty_publication f USING(professor_uid)
        LEFT JOIN reference_coverage c USING(work_id) GROUP BY a.professor_uid;
        UPDATE roster_coverage AS r SET
          n_linked=a.linked,n_supported=a.supported,n_refs=a.refs,
          n_supported_refs=a.supported_refs,n_with_topic=a.topics
        FROM affected_coverage a WHERE a.professor_uid=r.professor_uid;
        DELETE FROM recovery_queue
        WHERE reason IN ('no_supported_journal_link','journal_links_but_no_usable_references')
          AND professor_uid IN (SELECT professor_uid FROM affected_coverage);
        INSERT INTO recovery_queue
        SELECT r.professor_uid,r.subject,CASE WHEN r.n_supported=0 THEN 'no_supported_journal_link'
                                             ELSE 'journal_links_but_no_usable_references' END
        FROM roster_coverage r JOIN affected_coverage a USING(professor_uid)
        WHERE r.n_supported_refs=0;
    """)
    after = db.execute('SELECT COUNT(*) FROM faculty_publication').fetchone()[0]
    links, faculty = db.execute('SELECT count(*),count(DISTINCT professor_uid) FROM namesake_filter_audit').fetchone()
    assert before - after == links
    return {'removed_links': links, 'affected_faculty': faculty,
            'by_field': dict(db.execute('SELECT subject,count(*) FROM namesake_filter_audit '
                                       'JOIN roster_coverage USING(professor_uid) GROUP BY subject'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'koad_decisions', 'reference_cache', 'metadata_db', 'adjudications', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new private output SQLite file.')
    with closing(sqlite3.connect(read_only(args.reference_cache), uri=True)) as refs:
        decisions = candidates(args.koad_decisions, args.metadata_db, args.adjudications, canonicalizer(refs))
    with TemporaryDirectory(dir=args.output.parent) as directory:
        pending = Path(directory) / 'cohort.sqlite'
        with closing(sqlite3.connect(read_only(args.cohort), uri=True)) as original, \
                closing(sqlite3.connect(pending)) as target:
            original.backup(target)
            with target:
                summary = filter_cohort(target, decisions)
            assert target.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        args.output.hardlink_to(pending)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
