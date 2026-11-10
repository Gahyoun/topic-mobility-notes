#!/usr/bin/env python3
"""S10 frame stability in rotation-invariant terms (what T3 actually uses).

All T3 quantities (R, Q-hat, theta) are functions of inner products, so stability is measured on
  (a) pairwise cosine of random record pairs,
  (b) pairwise cosine of same-professor same-window pairs,
  (c) cell-level embedding Q-hat = 2n/(n-1) (1 - R^2) for professor x 2-year cells (n >= 2),
correlated between every space and the reference space. Cells use canonical records, de-duplicated within
a professor (removes the 199 same-paper-two-ids cases). Also compares cell embedding Q-hat with the
classification q_tree of S03 (different signal; expected moderate correlation).

  python3 -I scripts/s10_stability.py --ref b2025 --spaces b2010 b2015 b2020 b2025_s1 b2025_d256
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S10_stability"
rng = np.random.default_rng(11)


def load(tag):
    S = np.load(db.OUT / f"S08_space/space_{tag}.npz", allow_pickle=False)
    return pd.Series(np.arange(len(S["ids"])), index=S["ids"]), S["vectors"]


def cell_q(cells, pos, V):
    i = cells["record_id"].map(pos)
    ok = i.notna()
    c = cells[ok].assign(i=i[ok].astype(int))
    key = c["cell"].values
    uniq, inv = np.unique(key, return_inverse=True)
    S = np.zeros((len(uniq), V.shape[1]))
    np.add.at(S, inv, V[c["i"].values])
    n = np.bincount(inv).astype(float)
    R2 = (S ** 2).sum(1) / n ** 2
    q = np.where(n >= 2, 2 * n / np.maximum(n - 1, 1) * (1 - R2), np.nan)
    return pd.DataFrame({"cell": uniq, "n": n, "q_emb": q})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="b2025")
    ap.add_argument("--spaces", nargs="+", required=True)
    a = ap.parse_args()
    out = db.OUT / STEP; out.mkdir(parents=True, exist_ok=True)
    pos0, V0 = load(a.ref)
    wk = pd.read_csv(db.OUT / "S08_space/works_kept.csv.gz", usecols=["work_id", "record_id"])
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", usecols=["professor_uid", "work_id", "year", "subject"])
    prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv").set_index("professor_uid")["appointment_year"]
    rows = rows[rows.professor_uid.isin(prof.index) & (rows.year <= 2025)].merge(wk, on="work_id")
    rows["win"] = (rows.year - rows.professor_uid.map(prof).astype(int)) // 2
    rows["cell"] = rows.professor_uid + "|" + rows.win.astype(str)
    dup = rows.duplicated(["cell", "record_id"]).sum()
    cells = rows.drop_duplicates(["cell", "record_id"])

    common = pos0.index
    for t in a.spaces:
        common = common.intersection(load(t)[0].index)
    rp = rng.choice(len(common), (200_000, 2))
    rp = rp[rp[:, 0] != rp[:, 1]]
    ids_a, ids_b = common[rp[:, 0]], common[rp[:, 1]]
    sp = []
    for _, g in cells.groupby("cell")["record_id"]:
        u = g.values
        if len(u) >= 2:
            sp.append(rng.choice(u, 2, replace=False))
    sp = np.array(sp)
    sp = sp[np.isin(sp[:, 0], common) & np.isin(sp[:, 1], common)]

    def pc(pos, V, A, B):
        return np.einsum("ij,ij->i", V[pos[A].values], V[pos[B].values])

    ref_r, ref_s = pc(pos0, V0, ids_a, ids_b), pc(pos0, V0, sp[:, 0], sp[:, 1])
    ref_q = cell_q(cells, pos0, V0)
    res = []
    for t in a.spaces:
        pos, V = load(t)
        q = cell_q(cells, pos, V).merge(ref_q, on="cell", suffixes=("", "_ref")).dropna()
        res.append({"space": t, "r_random_pairs": np.corrcoef(pc(pos, V, ids_a, ids_b), ref_r)[0, 1],
                    "r_same_prof_pairs": np.corrcoef(pc(pos, V, sp[:, 0], sp[:, 1]), ref_s)[0, 1],
                    "r_cell_q": np.corrcoef(q.q_emb, q.q_emb_ref)[0, 1],
                    "r_cell_q_n>=5": np.corrcoef(q[q.n >= 5].q_emb, q[q.n >= 5].q_emb_ref)[0, 1],
                    "cells": len(q)})
    res = pd.DataFrame(res)
    # embedding Q vs classification q_tree (S03, w=2, P1)
    c3 = pd.read_csv(db.OUT / "S03_raoq/cells_w2.csv.gz")
    c3 = c3[c3.policy == "P1_decision"].assign(cell=lambda d: d.professor_uid + "|" + d.win.astype(str))
    m = ref_q.merge(c3[["cell", "q_tree", "q_topic", "subject"]], on="cell").dropna()
    ref_q.to_csv(out / f"cell_q_{a.ref}.csv.gz", index=False)
    res.to_csv(out / f"stability_vs_{a.ref}.csv", index=False)
    L = [f"# {STEP} (reference {a.ref})", "",
         f"- curve rows mapped to space records {len(rows):,}; duplicate record within a cell removed {dup:,}",
         f"- random pairs {len(ids_a):,}; same-professor-window pairs {len(sp):,}; cells with n>=2 {int((ref_q.n >= 2).sum()):,}",
         "", "Pearson correlation with the reference space:", "", res.round(3).to_markdown(index=False), "",
         f"embedding Q-hat vs classification q_tree (cells {len(m):,}): r = {np.corrcoef(m.q_emb, m.q_tree)[0, 1]:.3f}; "
         f"vs q_topic r = {np.corrcoef(m.q_emb, m.q_topic)[0, 1]:.3f}",
         f"embedding Q-hat distribution: median {ref_q.q_emb.median():.3f}, q10 {ref_q.q_emb.quantile(.1):.3f}, "
         f"q90 {ref_q.q_emb.quantile(.9):.3f} (2 = orthogonal on average)"]
    (out / f"stability_{a.ref}.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
