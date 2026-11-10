#!/usr/bin/env python3
"""S05 references: build work_references from the existing OpenAlex cache (no re-fetch).

Input : ~/Library/Caches/professor-career-dashboard/openalex/b*.jsonl.gz
        (one JSON per line: id, refs, abstract, keywords, topics, cited, venue; ids already short 'W…')
        requested id list: dashboard release work dir 01_work_ids.txt (204,611 ids)
Output: refs/work_references.sqlite (private, outside any repo; ATTACH it, never write into source DB)
  work_references(work_id, referenced_work_id)  PK(work_id, referenced_work_id)
  work_ref_status(work_id PK, status, n_refs, cited_by_count, shard)
     status in cached_with_refs | cached_no_refs | missing_in_openalex
  meta(key, value)
Kept / dropped / changes:
  S05_refs/changes.md, duplicate_records.csv (same id in >1 shard), self_refs_dropped.csv
"""
import csv, gzip, json, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S05_refs"


def main():
    out = db.OUT / STEP
    out.mkdir(parents=True, exist_ok=True)
    db.REFS_DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.REFS_DB.with_suffix(".sqlite.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE work_references(work_id TEXT NOT NULL, referenced_work_id TEXT NOT NULL,
                                     PRIMARY KEY(work_id, referenced_work_id)) WITHOUT ROWID;
        CREATE TABLE work_ref_status(work_id TEXT PRIMARY KEY, status TEXT NOT NULL, n_refs INTEGER,
                                     cited_by_count INTEGER, shard TEXT);
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    requested = (db.RELEASE_WORK / "01_work_ids.txt").read_text().split()
    req = set(requested)
    seen, dups, selfrefs, not_requested = {}, [], [], 0
    n_edges = n_dup_edges = 0
    shards = sorted(db.CACHE.glob("b*.jsonl.gz"))
    t0 = time.time()
    for sh in shards:
        batch_e, batch_s = [], []
        with gzip.open(sh, "rt", encoding="utf-8") as fh:
            for line in fh:
                w = json.loads(line)
                wid = db.short(w["id"])
                if wid in seen:
                    dups.append((wid, seen[wid], sh.name)); continue
                seen[wid] = sh.name
                if wid not in req:
                    not_requested += 1
                refs = []
                for r in w.get("refs") or []:
                    r = db.short(r)
                    if r == wid:
                        selfrefs.append((wid,)); continue
                    refs.append(r)
                u = set(refs)
                n_dup_edges += len(refs) - len(u)
                batch_e += [(wid, r) for r in u]
                batch_s.append((wid, "cached_with_refs" if u else "cached_no_refs", len(u),
                                w.get("cited"), sh.name))
        con.executemany("INSERT INTO work_references VALUES (?,?)", batch_e)
        con.executemany("INSERT INTO work_ref_status VALUES (?,?,?,?,?)", batch_s)
        n_edges += len(batch_e)
    missing = [w for w in requested if w not in seen]
    con.executemany("INSERT INTO work_ref_status VALUES (?, 'missing_in_openalex', NULL, NULL, NULL)",
                    [(w,) for w in missing])
    con.execute("CREATE INDEX idx_wr_ref ON work_references(referenced_work_id)")
    meta = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cache_dir": str(db.CACHE), "shards": len(shards),
            "requested_ids": len(requested), "rule": "dashboard release step 01: identity_decision=keep, KOAD "
            "namesake_paper excluded, article/review/letter/editorial, year<=2026 (no phd_year-5 filter)",
            "source": "OpenAlex works referenced_works, fetched 2026-11-04"}
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()
    st = dict(con.execute("SELECT status, COUNT(*) FROM work_ref_status GROUP BY 1").fetchall())
    con.close()
    tmp.replace(db.REFS_DB)

    with open(out / "duplicate_records.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("work_id", "first_shard", "dup_shard")] + dups)
    with open(out / "self_refs_dropped.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("work_id",)] + selfrefs)
    (out / "missing_in_openalex.txt").write_text("\n".join(missing) + "\n")
    cached = st.get("cached_with_refs", 0) + st.get("cached_no_refs", 0)
    L = [f"# {STEP}", "",
         f"- shards {len(shards):,}; records {len(seen):,} distinct (+{len(dups)} duplicate records dropped); "
         f"not in request list {not_requested}",
         f"- requested {len(requested):,}; cached {cached:,}; missing in OpenAlex {len(missing):,}",
         f"- status: {st}",
         f"- cached works with non-empty refs: {st.get('cached_with_refs', 0) / cached:.1%}; "
         f"of requested: {st.get('cached_with_refs', 0) / len(requested):.1%}",
         f"- edges {n_edges:,}; duplicate refs within a work collapsed {n_dup_edges:,}; self-refs dropped {len(selfrefs)}",
         f"- output {db.REFS_DB} ({db.REFS_DB.stat().st_size / 1e6:.0f} MB), {time.time() - t0:.0f}s"]
    (out / "changes.md").write_text("\n".join(L) + "\n")
    with open(db.OUT / "STEP_LOG.md", "a") as fh:
        fh.write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
