#!/usr/bin/env python3
"""S08 shared coordinate space (T2), same algorithm as bundle scale/scripts/space_fit.py.

Works  : distinct work_ids of S01 kept (main rule). Merged ids are mapped to their canonical record via
         refs.work_alias; a paper present under two DB ids enters the fit once (canonical record).
Features: referenced works shared by >= MIN_DF works of the population (refs channel only;
         citers channel not available — no cited-by lists in the cache).
Model  : X (binary) -> TfidfTransformer(sublinear_tf) -> TruncatedSVD(dims), fitted on records with
         year <= base_year, every record projected with the same transform, rows L2-normalised.
Diagnostics per record:
  captured = ||SVD(x)|| / ||x||   share of the TF-IDF norm the frame represents (projection quality)
  seen     = share of the record's features that occur in at least one fit row
Run for several base years; each run saves space_<tag>.npz + records_<tag>.csv.gz + changes.

  python3 -I scripts/s08_space.py --base-years 2010 2015 2020 0      (0 = all years <= 2025... fit on all)
"""
import argparse, json, sqlite3, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse.linalg  # noqa: F401
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S08_space"


def load(min_df):
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", usecols=["work_id", "year"])
    works = rows.groupby("work_id")["year"].min().rename("year").reset_index()
    con = sqlite3.connect(f"file:{db.REFS_DB}?mode=ro", uri=True)
    alias = dict(con.execute("SELECT requested_id, canonical_id FROM work_alias").fetchall())
    works["record_id"] = works["work_id"].map(lambda w: alias.get(w, w))
    st = pd.read_sql("SELECT work_id AS record_id, status FROM work_ref_status", con)
    works = works.merge(st, on="record_id", how="left")
    works["status"] = works["status"].fillna("not_requested")
    rec = works.groupby("record_id")["year"].min().rename("year").reset_index()
    E = pd.read_sql("SELECT work_id AS record_id, referenced_work_id AS ref FROM work_references", con)
    E = E[E["record_id"].isin(rec["record_id"])]
    df = E.groupby("ref").size()
    keep = df[df >= min_df].index
    E2 = E[E["ref"].isin(keep)]
    rec = rec[rec["record_id"].isin(E2["record_id"].unique())].reset_index(drop=True)
    ri = pd.Series(np.arange(len(rec)), index=rec["record_id"])
    fi = pd.Series(np.arange(len(keep)), index=keep)
    X = sparse.csr_matrix((np.ones(len(E2), np.float32), (ri[E2["record_id"]].values, fi[E2["ref"]].values)),
                          shape=(len(rec), len(keep)))
    info = {"db_work_ids": len(works), "records": int(works["record_id"].nunique()),
            "records_with_refs": int(E["record_id"].nunique()), "records_in_space": len(rec),
            "ref_vocabulary": int(len(df)), "features_min_df": int(len(keep)), "nnz": int(X.nnz),
            "dup_db_ids_same_record": int(len(works) - works["record_id"].nunique())}
    return works, rec, X, info


def fit(X, years, base_year, dims, seed=0):
    tf = TfidfTransformer(sublinear_tf=True)
    Xt = tf.fit_transform(X)
    svd = TruncatedSVD(n_components=dims, algorithm="randomized", n_iter=7, random_state=seed)
    m = years <= base_year
    svd.fit(Xt[m])
    V = svd.transform(Xt)
    xn = sparse.linalg.norm(Xt, axis=1)
    vn = np.linalg.norm(V, axis=1)
    seen_cols = np.asarray((X[m] > 0).sum(axis=0)).ravel() > 0
    nf = np.asarray(X.sum(axis=1)).ravel()
    seen = np.asarray(X[:, seen_cols].sum(axis=1)).ravel() / np.maximum(nf, 1)
    Vn = V / (vn[:, None] + 1e-12)
    return Vn.astype(np.float32), vn / np.maximum(xn, 1e-12), seen, float(svd.explained_variance_ratio_.sum()), int(m.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-years", type=int, nargs="+", default=[2010, 2015, 2020, 2025])
    p.add_argument("--dims", type=int, default=128)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = db.OUT / STEP; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    works, rec, X, info = load(a.min_df)
    years = rec["year"].values
    info["load_seconds"] = round(time.time() - t0)
    # kept / dropped at work level
    in_space = set(rec["record_id"])
    works["in_space"] = works["record_id"].isin(in_space)
    works["reason"] = np.where(works["in_space"], "",
                       np.where(works["status"].isin(["missing_in_openalex", "not_requested"]), "no_openalex_record",
                       np.where(works["status"].eq("cached_no_refs"), "empty_referenced_works",
                                "all_refs_below_min_df")))
    works[works["in_space"]].to_csv(out / "works_kept.csv.gz", index=False)
    works[~works["in_space"]].to_csv(out / "works_dropped.csv.gz", index=False)
    L = [f"# {STEP}", "", f"- {json.dumps(info)}",
         "- dropped DB work ids: " + ", ".join(f"{k} {v:,}" for k, v in works.loc[~works.in_space, 'reason'].value_counts().items()),
         f"- dims {a.dims}, min_df {a.min_df}, TF-IDF sublinear, TruncatedSVD randomized seed {a.seed}", "",
         "| base year | fit records | expl. var | captured median: fitted | 2016–20 | 2021–25 | seen median 2021–25 |",
         "|---|---|---|---|---|---|---|"]
    diag = []
    for by in a.base_years:
        V, cap, seen, ev, nfit = fit(X, years, by, a.dims, a.seed)
        tag = f"b{by}" + (f"_s{a.seed}" if a.seed else "") + (f"_d{a.dims}" if a.dims != 128 else "")
        np.savez_compressed(out / f"space_{tag}.npz", ids=rec["record_id"].values.astype(str), vectors=V,
                            years=years, dims=a.dims, base_year=by, channel="refs", min_df=a.min_df)
        r = pd.DataFrame({"record_id": rec["record_id"], "year": years, "captured": cap, "seen": seen})
        r.to_csv(out / f"records_{tag}.csv.gz", index=False)
        f = r[r.year <= by]; p1 = r[r.year.between(2016, 2020)]; p2 = r[r.year.between(2021, 2025)]
        L.append(f"| {by} | {nfit:,} | {ev:.3f} | {f.captured.median():.3f} | {p1.captured.median():.3f} | "
                 f"{p2.captured.median():.3f} | {p2.seen.median():.3f} |")
        yb = r.groupby((r.year // 5) * 5)[["captured", "seen"]].median()
        yb["base_year"] = by
        diag.append(yb.reset_index())
        print(L[-1], f"({time.time() - t0:.0f}s)", flush=True)
    pd.concat(diag).to_csv(out / "captured_by_year.csv", index=False)
    (out / "changes.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
