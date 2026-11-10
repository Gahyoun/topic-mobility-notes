"""Create a verified private SQLite archive for the study citation graph."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_network import load_inputs, citation_edges, ro, work_number, KEY, LEVELS
from freeze_sample import fingerprint

CUTOFF = '2025-12-31'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths','selection','sample-dir','private-root','edges','coordinates','analysis','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    cfg = json.loads(a.paths.read_text())
    sample = a.sample_dir
    diag = json.loads((a.analysis/'diagnostics.json').read_text())
    assert diag['mode'] == 'full'
    frozen = pd.read_csv(sample/'frozen_rows.csv.gz',dtype={col:str for col in LEVELS})
    targets = pd.read_csv(sample/'graph_targets.csv.gz')
    assert fingerprint(frozen) == diag['original_frozen_sha256'] == diag['matched_sample_sha256']
    assert diag['publications']==frozen.id.nunique() and diag['windows']==frozen.groupby(KEY).ngroups
    with ro(a.private_root/'refs/work_references.sqlite') as db:
        cache_status = {int(w[1:]):status for w,status in db.execute("SELECT work_id,status FROM work_ref_status")}
        old_cache = {w for w,status in cache_status.items() if status != 'missing_in_openalex'}
        aliases = {int(k[1:]):int(v[1:]) for k,v in db.execute('SELECT requested_id,canonical_id FROM work_alias')}
        old_source = dict(db.execute('SELECT key,value FROM meta WHERE key IN (\'built_at\',\'source\')'))
    _, ref_ids, ref_vectors, _, local_edges, years, _ = load_inputs(cfg,a.private_root,a.selection)
    target_ids = pd.Index(targets.id)
    refs, local_in, _ = citation_edges(local_edges,target_ids,years)
    archive_years = years.reindex(sorted(old_cache)).dropna()
    archive_years = archive_years.loc[archive_years.le(2025)].astype(int)
    full_local_refs = local_edges.loc[local_edges.citing.isin(archive_years.index), ['citing','cited']]
    assert refs.citing.isin(archive_years.index).all()
    faculty = pd.read_csv(a.selection,dtype={col:str for col in LEVELS})
    faculty = faculty.loc[faculty.year.le(2025)].copy()
    faculty['raw'] = work_number(faculty.work_id)
    faculty['id'] = faculty.raw.map(aliases).fillna(faculty.raw).astype('int64')
    faculty = faculty.assign(native=faculty.raw.eq(faculty.id)).sort_values(
        ['native','year','id'],ascending=[False,True,True],kind='stable')
    faculty = faculty.drop_duplicates(['professor_uid','id'])
    faculty_works = faculty.drop_duplicates('id')[['id','year']].copy()
    faculty_works['year'] = faculty_works.id.map(archive_years).fillna(faculty_works.year).astype(int)
    faculty_works['reference_status'] = faculty_works.id.map(cache_status).fillna('not_in_cache')
    assert targets.id.isin(faculty_works.id).all()
    faculty_rows = faculty[['professor_uid','subject','id','year'] + LEVELS]
    faculty_rows = faculty_rows.astype(object).where(faculty_rows.notna(),None)

    fresh = pd.read_csv(a.edges)
    fresh['citing'] = fresh.citing.map(aliases).fillna(fresh.citing).astype('int64')
    fresh = fresh.drop_duplicates(['citing','cited'])
    assert fresh.cited.isin(target_ids).all()
    assert (fresh.citing_year-fresh.cited.map(years)).between(1,3).all()
    assert fresh.citing_year.max()<=2025
    fresh['new_external'] = (~fresh.citing.isin(old_cache) & ~fresh.citing.isin(target_ids)).astype(int)
    assert fresh.new_external.sum()==diag['external_links']
    local_in['citing_year'] = local_in.citing.map(years).astype(int)
    assert local_in.citing_year.max()<=2025 and targets.year.max()<=2022

    db_path = a.out/('openalex-topic-mobility-through-'+CUTOFF+'.sqlite')
    zip_path = db_path.with_suffix('.sqlite.gz')
    if db_path.exists() or zip_path.exists():
        raise FileExistsError('An archive already exists at the destination')
    api_completed = datetime.fromtimestamp(a.edges.with_suffix('.json').stat().st_mtime,timezone.utc)
    meta = {
        'archive_scope':'Korean faculty topic mobility study citation graph, not a historical OpenAlex snapshot',
        'publication_cutoff':CUTOFF,
        'openalex_collection_completed_utc':api_completed.isoformat(timespec='seconds'),
        'openalex_endpoint':'https://api.openalex.org/works',
        'assembled_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'local_reference_cache':old_source,
        'incoming_exposure':'Publication year +1 through +3 inclusive',
        'id_convention':'Numeric suffix of the canonical OpenAlex work ID, prefix W when displaying',
        'vector_encoding':'Little-endian float32, raw C-order bytes; dim given per embedding',
        'frozen_matched_sample_sha256':fingerprint(frozen),
        'diagnostics':diag,
    }
    with sqlite3.connect(db_path) as db:
        db.executescript("""
          PRAGMA journal_mode=OFF;
          PRAGMA user_version=1;
          CREATE TABLE metadata (key TEXT PRIMARY KEY, json_value TEXT NOT NULL) WITHOUT ROWID;
          CREATE TABLE graph_work (work_id INTEGER PRIMARY KEY, publication_year INTEGER NOT NULL CHECK(publication_year <= 2022));
          CREATE TABLE matched_panel (professor_uid TEXT, subject TEXT NOT NULL, window_index INTEGER,
              work_id INTEGER REFERENCES graph_work, publication_year INTEGER NOT NULL CHECK(publication_year <= 2022),
              topic_id TEXT NOT NULL, subfield_id TEXT NOT NULL, field_id TEXT NOT NULL, domain_id TEXT NOT NULL,
              PRIMARY KEY(professor_uid,window_index,work_id)) WITHOUT ROWID;
          CREATE TABLE faculty_work (work_id INTEGER PRIMARY KEY, publication_year INTEGER NOT NULL CHECK(publication_year <= 2025),
              reference_status TEXT NOT NULL);
          CREATE TABLE faculty_publication (professor_uid TEXT, subject TEXT NOT NULL,
              work_id INTEGER REFERENCES faculty_work, publication_year INTEGER NOT NULL CHECK(publication_year <= 2025),
              topic_id TEXT, subfield_id TEXT, field_id TEXT, domain_id TEXT,
              PRIMARY KEY(professor_uid,work_id)) WITHOUT ROWID;
          CREATE TABLE local_work (work_id INTEGER PRIMARY KEY, publication_year INTEGER NOT NULL CHECK(publication_year <= 2025));
          CREATE TABLE local_reference (citing_id INTEGER REFERENCES local_work, cited_id INTEGER,
              PRIMARY KEY(citing_id,cited_id)) WITHOUT ROWID;
          CREATE VIEW outgoing_reference AS
              SELECT r.citing_id,r.cited_id FROM local_reference r JOIN graph_work w ON w.work_id=r.citing_id;
          CREATE TABLE local_incoming (cited_id INTEGER REFERENCES graph_work, citing_id INTEGER,
              citing_year INTEGER NOT NULL CHECK(citing_year <= 2025), PRIMARY KEY(cited_id,citing_id)) WITHOUT ROWID;
          CREATE TABLE openalex_incoming (cited_id INTEGER REFERENCES graph_work, citing_id INTEGER,
              citing_year INTEGER NOT NULL CHECK(citing_year <= 2025), new_external INTEGER NOT NULL CHECK(new_external IN (0,1)),
              PRIMARY KEY(cited_id,citing_id)) WITHOUT ROWID;
          CREATE VIEW faculty_outgoing AS
              SELECT r.citing_id,r.cited_id FROM local_reference r JOIN faculty_work w ON w.work_id=r.citing_id;
          CREATE VIEW faculty_incoming_local_to_cutoff AS
              SELECT r.cited_id,r.citing_id,w.publication_year AS citing_year
              FROM local_reference r JOIN faculty_work f ON f.work_id=r.cited_id JOIN local_work w ON w.work_id=r.citing_id;
          CREATE VIEW faculty_incoming_openalex_3yr AS
              SELECT i.cited_id,i.citing_id,i.citing_year,i.new_external
              FROM openalex_incoming i JOIN faculty_work f ON f.work_id=i.cited_id;
          CREATE TABLE embedding (work_id INTEGER REFERENCES graph_work, model TEXT, dimensions INTEGER NOT NULL,
              vector BLOB NOT NULL, CHECK(length(vector)=4*dimensions), PRIMARY KEY(work_id,model)) WITHOUT ROWID;
        """)
        db.execute('PRAGMA foreign_keys=ON')
        db.executemany('INSERT INTO metadata VALUES (?,?)', ((k,json.dumps(v,ensure_ascii=False)) for k,v in meta.items()))
        db.executemany('INSERT INTO graph_work VALUES (?,?)',targets[['id','year']].itertuples(index=False,name=None))
        db.executemany('INSERT INTO matched_panel VALUES (?,?,?,?,?,?,?,?,?)',
                       frozen[KEY+['id','year']+LEVELS].itertuples(index=False,name=None))
        db.executemany('INSERT INTO faculty_work VALUES (?,?,?)',faculty_works.itertuples(index=False,name=None))
        db.executemany('INSERT INTO faculty_publication VALUES (?,?,?,?,?,?,?,?)',faculty_rows.itertuples(index=False,name=None))
        db.executemany('INSERT INTO local_work VALUES (?,?)',((int(w),int(y)) for w,y in archive_years.items()))
        db.executemany('INSERT INTO local_reference VALUES (?,?)',full_local_refs.itertuples(index=False,name=None))
        db.executemany('INSERT INTO local_incoming VALUES (?,?,?)',local_in[['cited','citing','citing_year']].itertuples(index=False,name=None))
        db.executemany('INSERT INTO openalex_incoming VALUES (?,?,?,?)',
                       fresh[['cited','citing','citing_year','new_external']].itertuples(index=False,name=None))
        ref_idx=ref_ids.get_indexer(target_ids)
        assert (ref_idx>=0).all()

        def add_vectors(ids, vectors, model):
            rows=((int(w),model,len(v),np.asarray(v,dtype='<f4').tobytes()) for w,v in zip(ids,vectors,strict=True))
            db.executemany('INSERT INTO embedding VALUES (?,?,?,?)',rows)

        add_vectors(target_ids,ref_vectors[ref_idx],'shared_references_b2025')
        with np.load(a.coordinates) as saved:
            ids=saved['ids']
            assert pd.Index(ids).is_unique and pd.Index(ids).isin(target_ids).all()
            for model in ('with_external','local','fixed_pivots'):
                add_vectors(ids,saved[model],'citation_'+model)
        db.execute('CREATE INDEX local_reference_by_cited ON local_reference(cited_id,citing_id)')
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
        assert db.execute('PRAGMA foreign_key_check').fetchone() is None
        for table in ('faculty_work','faculty_publication','faculty_outgoing','faculty_incoming_local_to_cutoff',
                      'faculty_incoming_openalex_3yr','graph_work','matched_panel','local_work','local_reference',
                      'outgoing_reference','local_incoming','openalex_incoming','embedding'):
            meta.setdefault('archive_rows',{})[table]=db.execute('SELECT count(*) FROM '+table).fetchone()[0]
        assert meta['archive_rows']['matched_panel']==len(frozen)
        assert meta['archive_rows']['outgoing_reference']==len(refs)
        assert meta['archive_rows']['openalex_incoming']==len(fresh)
        for table in ('local_incoming','openalex_incoming'):
            invalid=db.execute(f"""SELECT count(*) FROM {table} i JOIN graph_work g ON g.work_id=i.cited_id
                                WHERE i.citing_year-g.publication_year NOT BETWEEN 1 AND 3""").fetchone()[0]
            assert invalid==0
    expected = sha256(db_path)
    with db_path.open('rb') as f, gzip.GzipFile(filename=zip_path,mode='wb',compresslevel=6,mtime=0) as z:
        shutil.copyfileobj(f,z,length=1024*1024)
    with gzip.open(zip_path,'rb') as z:
        assert hashlib.file_digest(z,'sha256').hexdigest() == expected
    (a.out/'archive-manifest.json').write_text(json.dumps(meta|{'database_sha256':expected,
        'gzip_sha256':sha256(zip_path),'database_bytes':db_path.stat().st_size,
        'gzip_bytes':zip_path.stat().st_size},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta['archive_rows'],indent=2))
    print(zip_path)


if __name__=='__main__':
    main()
