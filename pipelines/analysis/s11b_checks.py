#!/usr/bin/env python3
"""S11b checks for the panel quantities (no new data; recomputes from the space).

1. q_hat equals the brute-force mean pairwise squared distance (200 cells).
2. Integration identity: with explicit leave-one-out reference vectors y_r, within + between == total and
   within equals 1 - ||c_p||^2 averaged (50 cells).
3. theta noise-floor calibration under a no-shift null: each cell with n >= 8 is split at random into two
   pseudo-windows; theta between them vs theta0 from half-splits of each pseudo-window.
   Calibrated if mean(theta^2) / mean(theta0^2) ~ 1.

  python3 -I analysis/s11b_checks.py --tag b2025
"""
import argparse, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from s11_panel import angle, ref_coordinates, K_SPLIT  # noqa: E402

rng = np.random.default_rng(5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="b2025")
    a = ap.parse_args()
    out = db.OUT / "S11_panel"
    S = np.load(db.OUT / f"S08_space/space_{a.tag}.npz")
    pos = pd.Series(np.arange(len(S["ids"])), index=S["ids"]); V = S["vectors"].astype(np.float64)
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", usecols=["professor_uid", "work_id", "year"])
    prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv").set_index("professor_uid")["appointment_year"]
    wk = pd.read_csv(db.OUT / "S08_space/works_kept.csv.gz", usecols=["work_id", "record_id"])
    rows = rows[rows.professor_uid.isin(prof.index) & (rows.year <= 2025)].merge(wk, on="work_id")
    rows = rows.drop_duplicates(["professor_uid", "record_id"])
    rows["i"] = pos[rows.record_id].values
    rows["win"] = (rows.year - rows.professor_uid.map(prof).astype(int)) // 2
    cells = [g["i"].values for _, g in rows.groupby(["professor_uid", "win"]) if len(g) >= 2]
    P = pd.read_csv(out / f"panel_{a.tag}_P1.csv.gz")
    L = [f"# S11b checks ({a.tag})", ""]

    # 1. q_hat
    err = 0.0
    for k in rng.choice(len(cells), 200, replace=False):
        X = V[cells[k]]; n = len(X)
        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        qb = D.sum() / (n * (n - 1))
        m = X.mean(0); qf = 2 * n / (n - 1) * (1 - m @ m)
        err = max(err, abs(qb - qf))
    L.append(f"1. q_hat vs brute force, 200 cells: max abs diff {err:.2e}")

    # 2. integration identity with explicit y_r
    C, nref = ref_coordinates(pos, V.astype(np.float32), int(S["min_df"]))
    con = sqlite3.connect(f"file:{db.REFS_DB}?mode=ro", uri=True)
    E = pd.read_sql("SELECT work_id AS rid, referenced_work_id AS ref FROM work_references", con)
    E = E[E.rid.isin(pos.index)]
    df = E.groupby("ref").size(); E = E[E.ref.map(df) >= int(S["min_df"])]
    E["i"] = pos[E.rid].values
    citers = E.groupby("ref")["i"].apply(np.array)
    by_paper = E.groupby("i")["ref"].apply(list)
    worst = worst_c = 0.0
    for k in rng.choice(len(cells), 50, replace=False):
        idx = [i for i in cells[k] if i in by_paper.index]
        if len(idx) < 2:
            continue
        Ys, cps = [], []
        for p in idx:
            Y = []
            for r in by_paper[p]:
                q = citers[r]; s = V[q].sum(0) - V[p]
                Y.append(s / (np.linalg.norm(s) + 1e-12))
            Y = np.array(Y); Ys.append(Y); cps.append(Y.mean(0))
        cps = np.array(cps); cbar = cps.mean(0)
        within = np.mean([np.mean(((Y - cp) ** 2).sum(1)) for Y, cp in zip(Ys, cps)])
        between = np.mean(((cps - cbar) ** 2).sum(1))
        total = np.mean([np.mean(((Y - cbar) ** 2).sum(1)) for Y in Ys])
        worst = max(worst, abs(within + between - total))
        worst_c = max(worst_c, float(np.abs(cps - C[idx]).max()))
    L.append(f"2. integration: max |within + between - total| {worst:.2e}; max |c_p explicit - vectorised| {worst_c:.2e} (50 cells)")

    # 3. theta noise floor calibration
    th2, t02, u = [], [], []
    for c in cells:
        if len(c) < 8:
            continue
        p = rng.permutation(c); k = len(p) // 2
        A, B = V[p[:k]], V[p[k:]]
        h = []
        for Z in (A, B):
            s = [angle(Z[q[:len(Z) // 2]].mean(0), Z[q[len(Z) // 2:]].mean(0)) ** 2
                 for q in (rng.permutation(len(Z)) for _ in range(K_SPLIT))]
            h.append(np.mean(s))
        th2.append(angle(A.mean(0), B.mean(0)) ** 2); t02.append(h[0] / 4 + h[1] / 4)
        ma, mb = A.mean(0), B.mean(0); na, nb = len(A), len(B)
        u.append((na * ma @ ma - 1) / (na - 1) + (nb * mb @ mb - 1) / (nb - 1) - 2 * ma @ mb)
    th2, t02, u = np.array(th2), np.array(t02), np.array(u)
    L.append(f"3. no-shift null, {len(th2):,} cells (n >= 8): mean theta^2 {th2.mean():.4f}, mean theta0^2 {t02.mean():.4f}, "
             f"ratio {th2.mean() / t02.mean():.3f}; share theta_corr == 0 {(th2 <= t02).mean():.1%}")
    se = u.std() / np.sqrt(len(u))
    L.append(f"   shift2_u under the same null: mean {u.mean():+.5f} (SE {se:.5f}); real panel mean {P.shift2_u.mean():+.4f}, "
             f"median {P.shift2_u.median():+.4f}")
    d = P[P.theta_corr.notna()]
    L.append(f"   real panel: cells with theta_corr {len(d):,}; at the floor (== 0) {(d.theta_corr == 0).mean():.1%}; "
             f"mean theta^2 {np.mean(d.theta ** 2):.4f} vs mean theta0^2 {np.mean(d.theta0 ** 2):.4f}")
    (out / f"checks_{a.tag}.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
