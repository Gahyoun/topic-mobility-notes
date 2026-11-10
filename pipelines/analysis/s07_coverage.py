#!/usr/bin/env python3
"""S07 coverage of referenced_works for the analysis population, and in-scope works absent from the cache.

Populations:
  A  S01 kept rows (main rule incl. phd_year-5, journal types)       -> acceptance check (>= 80 %)
  A' A restricted to curve rows (appointment year known, year <= 2025)
  N  rows passing every rule except journal type (non-journal types) -> not requested, listed only
Coverage = share with non-empty refs in work_references (via ATTACH; source DB opened read-only).
Outputs: S07_coverage/coverage.md, coverage_by_year.csv, coverage_by_subject.csv,
         professors_low_coverage.csv, not_in_cache_journal.csv, not_in_cache_nonjournal.csv
"""
import sqlite3, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

out = db.OUT / "S07_coverage"; out.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(f"file:{db.REFS_DB}?mode=ro", uri=True)
status = pd.read_sql("SELECT work_id, status, n_refs FROM work_ref_status", con)
alias = pd.read_sql("SELECT requested_id, canonical_id FROM work_alias", con)
st = status.set_index("work_id")
amap = dict(zip(alias.requested_id, alias.canonical_id))

def resolve(df):
    df = df.copy()
    df["status"] = df["work_id"].map(st["status"]).fillna("not_requested")
    df["n_refs"] = df["work_id"].map(st["n_refs"])
    # merged ids: use canonical record when unambiguously mapped
    m = df["status"].eq("missing_in_openalex") & df["work_id"].isin(amap)
    can = df.loc[m, "work_id"].map(amap)
    df.loc[m, "status"] = can.map(st["status"]).fillna("missing_in_openalex").radd("alias:")
    df.loc[m, "n_refs"] = can.map(st["n_refs"])
    df["has_refs"] = df["n_refs"].fillna(0) > 0
    return df

dt = {"topic_id": str, "subfield_id": str, "field_id": str, "domain_id": str}
A = resolve(pd.read_csv(db.OUT / "S01_select/kept.csv.gz", dtype=dt))
prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv")
Ac = A[A.professor_uid.isin(prof.professor_uid) & (A.year <= 2025)]
W = A.drop_duplicates("work_id")
L = ["# S07_coverage", "",
     "| population | units | with refs | share |", "|---|---|---|---|"]
for name, df in [("A distinct works", W), ("A professor-paper rows", A), ("A' curve rows", Ac)]:
    L.append(f"| {name} | {len(df):,} | {int(df.has_refs.sum()):,} | {df.has_refs.mean():.1%} |")
L += ["", "status of A distinct works: " + ", ".join(f"{k} {v:,}" for k, v in W.status.value_counts().items())]
L.append(f"acceptance (>= 80 % of distinct works with refs): {'PASS' if W.has_refs.mean() >= 0.8 else 'FAIL'}")
yr = W.groupby(pd.cut(W.year, [0, 1994, 1999, 2004, 2009, 2014, 2019, 2023, 2026]), observed=True).has_refs.agg(["size", "mean"])
yr.to_csv(out / "coverage_by_year.csv")
sj = A.groupby("subject").has_refs.agg(["size", "mean"]); sj.to_csv(out / "coverage_by_subject.csv")
wt = W.groupby("work_type").has_refs.agg(["size", "mean"])
pc = A.groupby(["professor_uid", "subject"]).has_refs.agg(n="size", share="mean").reset_index()
low = pc[pc["share"] < 0.8]; low.to_csv(out / "professors_low_coverage.csv", index=False)
L += ["", "by publication-year bin (distinct works): " + "; ".join(f"{k} {r['mean']:.1%} (n={int(r['size']):,})" for k, r in yr.iterrows()),
      "by subject (rows): " + "; ".join(f"{k} {r['mean']:.1%}" for k, r in sj.iterrows()),
      "by work_type (distinct works): " + "; ".join(f"{k} {r['mean']:.1%} (n={int(r['size']):,})" for k, r in wt.iterrows()),
      f"professors with < 80 % coverage: {len(low):,} of {len(pc):,} (median share {pc['share'].median():.1%})"]
# in-scope works not in the cache
miss_j = W[W.status.isin(["missing_in_openalex", "alias:missing_in_openalex", "not_requested"])][["work_id", "year", "work_type", "status"]]
miss_j.to_csv(out / "not_in_cache_journal.csv", index=False)
D = pd.read_csv(db.OUT / "S01_select/dropped.csv.gz", dtype=dt)
N = D[(D.reason == "non_journal_type") & D.year.notna() & (D.year <= db.MAX_YEAR)]
N = N.merge(prof[["professor_uid", "phd_year"]], on="professor_uid", how="left")
N = N[~(N.phd_year.notna() & (N.year < N.phd_year - 5))]
Nw = N.drop_duplicates("work_id")
Nw = Nw[~Nw.work_id.isin(st.index)]
Nw[["work_id", "year", "work_type"]].to_csv(out / "not_in_cache_nonjournal.csv", index=False)
L += ["", f"in-scope journal works without a cached record: {len(miss_j):,} (status: {miss_j.status.value_counts().to_dict()})",
      f"non-journal works passing all other rules, never requested: {len(Nw):,} distinct ({len(N):,} rows); by type: "
      + ", ".join(f"{k} {v:,}" for k, v in Nw.work_type.value_counts().items()),
      f"  -> fetch cost if approved: ~{-(-len(Nw) // 100):,} calls of 100 ids"]
(out / "coverage.md").write_text("\n".join(L) + "\n")
open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
print("\n".join(L))
