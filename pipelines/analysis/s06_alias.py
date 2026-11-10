#!/usr/bin/env python3
"""S06 alias: recover requested-id -> canonical OpenAlex id for merged works, batch by batch.

OpenAlex answers a filter on a merged (old) id with the canonical record, without saying which
requested id it answers. Within one 100-id batch: missing = requested - returned, extra = returned - requested.
When a batch has exactly as many missing as extra and both are 1, the pair is unambiguous.
Ambiguous batches are saved, not guessed. Table work_alias(requested_id, canonical_id, method) goes into
refs/work_references.sqlite.  Outputs: S06_alias/alias_kept.csv, alias_ambiguous.csv, changes.md
"""
import csv, gzip, json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

out = db.OUT / "S06_alias"; out.mkdir(parents=True, exist_ok=True)
req = (db.RELEASE_WORK / "01_work_ids.txt").read_text().split()
reqset = set(req)
pairs, amb, truly_missing = [], [], []
for b in range(0, len(req), 100):
    asked = req[b:b + 100]
    sh = db.CACHE / f"b{b // 100:05d}.jsonl.gz"
    got = [json.loads(l)["id"] for l in gzip.open(sh, "rt")] if sh.exists() else []
    miss = [w for w in asked if w not in set(got)]
    extra = [w for w in got if w not in set(asked)]
    if len(miss) == 1 and len(extra) == 1:
        pairs.append((miss[0], extra[0], "single_pair_in_batch"))
    elif miss and extra:
        amb.append((sh.name, " ".join(miss), " ".join(extra)))
    elif miss:
        truly_missing += miss
with open(out / "alias_kept.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("requested_id", "canonical_id", "method")] + pairs)
with open(out / "alias_ambiguous.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("shard", "missing_ids", "extra_ids")] + amb)
(out / "no_record.txt").write_text("\n".join(truly_missing) + "\n")
con = sqlite3.connect(db.REFS_DB)
con.execute("DROP TABLE IF EXISTS work_alias")
con.execute("CREATE TABLE work_alias(requested_id TEXT PRIMARY KEY, canonical_id TEXT NOT NULL, method TEXT)")
con.executemany("INSERT INTO work_alias VALUES (?,?,?)", pairs)
con.commit()
both_req = sum(c in reqset for _, c, _ in pairs)
amb_miss = sum(len(a[1].split()) for a in amb)
L = ["# S06_alias", "",
     f"- unambiguous merged-id pairs {len(pairs):,} (canonical id also requested, i.e. same paper twice in DB: {both_req:,})",
     f"- ambiguous batches {len(amb)} covering {amb_miss} missing ids (not mapped)",
     f"- requested ids with no returned record in their batch and no candidate: {len(truly_missing):,}",
     "- table work_alias written to refs/work_references.sqlite"]
(out / "changes.md").write_text("\n".join(L) + "\n")
open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
print("\n".join(L))
