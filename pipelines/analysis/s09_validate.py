#!/usr/bin/env python3
"""S09 validate the chosen space with signals not used to build it, and measure frame dependence.

1. Hierarchy: cosine of random record pairs by OpenAlex relation (same topic / same subfield /
   same field / same domain / different domain). Caveat: OpenAlex topic classifier also uses citations,
   so this is only partly independent of the reference channel.
2. Authorship: pairs of papers by the same professor in the same 2-year window vs random pairs
   with the same subfield (controls for classification) — authorship is not used in construction.
3. Frame dependence: orthogonal Procrustes of the chosen space onto every other base-year space;
   per-record cosine after alignment, median by 5-year publication bin (handoff §4: report residual).

  python3 -I scripts/s09_validate.py --tag b2015
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S09_validate"
rng = np.random.default_rng(7)


def load_space(tag):
    S = np.load(db.OUT / f"S08_space/space_{tag}.npz", allow_pickle=False)
    return pd.Series(np.arange(len(S["ids"])), index=S["ids"]), S["vectors"], S["years"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    ap.add_argument("--others", nargs="+", default=["b2010", "b2015", "b2020", "b2025"])
    a = ap.parse_args()
    out = db.OUT / STEP; out.mkdir(parents=True, exist_ok=True)
    pos, V, years = load_space(a.tag)
    wk = pd.read_csv(db.OUT / "S08_space/works_kept.csv.gz", usecols=["work_id", "record_id"])
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", dtype=str)
    rows = rows.merge(wk, on="work_id")
    rows["i"] = rows["record_id"].map(pos)
    lev = rows.drop_duplicates("record_id").set_index("i")[["topic_id", "subfield_id", "field_id", "domain_id"]].sort_index()
    T = lev.reindex(np.arange(len(V)))

    # 1. hierarchy
    n = 2_000_000
    i, j = rng.integers(0, len(V), n), rng.integers(0, len(V), n)
    m = i != j; i, j = i[m], j[m]
    cos = np.einsum("ij,ij->i", V[i], V[j])
    rel = np.full(len(i), "5 different domain", dtype=object)
    for lvl, name in [("domain_id", "4 same domain"), ("field_id", "3 same field"),
                      ("subfield_id", "2 same subfield"), ("topic_id", "1 same topic")]:
        rel[T[lvl].values[i] == T[lvl].values[j]] = name
    H = pd.DataFrame({"rel": rel, "cos": cos}).groupby("rel")["cos"].agg(["size", "mean", "median"])
    # extra same-topic pairs (rare under uniform sampling)
    g = pd.Series(np.arange(len(V)), index=T["topic_id"].values)
    tp = [rng.choice(ix, 2, replace=False) for _, ix in g.groupby(level=0) if len(ix) >= 2 for _ in range(3)]
    tp = np.array(tp)
    H.loc["1 same topic (stratified)"] = [len(tp), np.mean(np.einsum("ij,ij->i", V[tp[:, 0]], V[tp[:, 1]])),
                                          np.median(np.einsum("ij,ij->i", V[tp[:, 0]], V[tp[:, 1]]))]

    # 2. authorship
    prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv").set_index("professor_uid")["appointment_year"]
    r = rows[rows["professor_uid"].isin(prof.index)].copy()
    r["win"] = (r["year"].astype(int) - r["professor_uid"].map(prof).astype(int)) // 2
    pairs = []
    for _, ix in r.groupby(["professor_uid", "win"])["i"]:
        ix = ix.dropna().astype(int).unique()
        if len(ix) >= 2:
            k = min(3, len(ix) * (len(ix) - 1) // 2)
            for _ in range(k):
                pairs.append(rng.choice(ix, 2, replace=False))
    pairs = np.array(pairs)
    same_prof = np.einsum("ij,ij->i", V[pairs[:, 0]], V[pairs[:, 1]])
    sf = T["subfield_id"].values[pairs[:, 0]]
    gsf = pd.Series(np.arange(len(V)), index=T["subfield_id"].values).groupby(level=0).apply(lambda s: s.values)
    ctrl = np.array([rng.choice(gsf[s]) for s in sf])
    ctrl_cos = np.einsum("ij,ij->i", V[pairs[:, 0]], V[ctrl])
    same_sf_prof = T["subfield_id"].values[pairs[:, 0]] == T["subfield_id"].values[pairs[:, 1]]
    A = {"same_prof_window_pairs": len(pairs), "same_prof_mean": float(same_prof.mean()),
         "same_prof_same_subfield_mean": float(same_prof[same_sf_prof].mean()),
         "random_same_subfield_mean": float(ctrl_cos.mean()),
         "auc_same_prof_vs_same_subfield_random": float((same_prof[same_sf_prof][:, None] >
                                                         ctrl_cos[rng.integers(0, len(ctrl_cos), 2000)][None, :]).mean())}

    # 3. Procrustes
    P = []
    for o in a.others:
        if o == a.tag:
            continue
        pos2, V2, _ = load_space(o)
        common = pos.index.intersection(pos2.index)
        X, Y = V[pos[common].values], V2[pos2[common].values]
        R, _ = orthogonal_procrustes(X, Y)
        c = np.einsum("ij,ij->i", X @ R, Y)
        yb = (years[pos[common].values] // 5) * 5
        s = pd.Series(c).groupby(yb).median()
        P.append(pd.Series({"vs": o, "median_all": float(np.median(c)), **{f"{int(k)}": v for k, v in s.items()}}))
    P = pd.DataFrame(P)

    H.to_csv(out / f"hierarchy_{a.tag}.csv"); P.to_csv(out / f"procrustes_{a.tag}.csv", index=False)
    L = [f"# {STEP} ({a.tag})", "", "random-pair cosine by OpenAlex relation:", "", H.round(3).to_markdown(), "",
         "authorship (not used in construction): " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v:,}"
                                                             for k, v in A.items()), "",
         "Procrustes-aligned per-record cosine vs other base years (median, by 5-year bin):", "",
         P.round(3).to_markdown(index=False)]
    (out / f"validate_{a.tag}.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
