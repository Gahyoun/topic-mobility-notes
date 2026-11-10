#!/usr/bin/env python3
"""S11 person-window panel (T3) on the shared space (main: b2025).

One row = one professor x event-time window (w years, from S02). Papers: S01 kept (KOAD policy P1, or P3 with
--policy P3), year <= 2025, mapped to canonical space records, de-duplicated within professor.

Continuous quantities on unit vectors x_p (handoff §3):
  m = mean x_p, R = ||m||, c = m / R
  q_hat   = 2n/(n-1) (1 - R^2)              Rao's Q-hat (U-statistic, unbiased for any n >= 2)
  r2_corr = (n R^2 - 1)/(n - 1)             bias-corrected concentration R~^2; d_corr = 1 - sqrt(max(0, R~^2))
  theta   = arccos(c_t . c_prev)            shift between consecutive active windows (gap recorded)
  theta0  = noise floor: mean over K random half-splits of each cell of angle(halfA, halfB)^2 / 4,
            summed over the two cells   [centroid-direction variance ~ s^2/n; two halves of n/2 differ by 4 s^2/n,
            two full cells by s_t^2/n_t + s_prev^2/n_prev]
  theta_corr = sqrt(max(0, theta^2 - theta0^2))   (needs n >= 4 in both cells)
  persistence = cos between consecutive shift vectors c_t - c_prev
  shift2_u = R~^2_t + R~^2_prev - 2 m_t . m_prev   unbiased U-statistic for ||mu_t - mu_prev||^2 (no noise floor
            needed; expectation 0 when nothing moved). Preferred over theta_corr: the half-split floor is
            ~19 % too small at the observed angles (S11b check 3).
Integration ratio I (two-track vs integrated): references r of paper p get coordinates
  y_r^(-p) = normalise( sum_{q cites r, q != p} x_q )   (leave-one-out centroid of the other citing papers)
  c_p = mean_r y_r^(-p);  per cell, with weights 1/n:
  within = mean_p (1 - ||c_p||^2), between = mean_p ||c_p||^2 - ||cbar||^2, total = 1 - ||cbar||^2 (exact identity)
  integ = within / total
Discrete visibility quantities (not interdisciplinarity): MiniBatchKMeans k clusters on the space,
  hill0, hill1, entropy_chao on cluster counts, n_clusters, cluster_jump, top_cluster.

Outputs (private): S11_panel/panel_<tag>_<policy>.csv.gz, dropped_<tag>_<policy>.csv.gz, changes.md
  python3 -I analysis/s11_panel.py --tag b2025 --policy P1
"""
import argparse, json, sqlite3, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S11_panel"
K_SPLIT = 20


def ref_coordinates(pos, V, min_df):
    """Per record c_p = mean over its refs of leave-one-out normalised citing-paper centroids."""
    con = sqlite3.connect(f"file:{db.REFS_DB}?mode=ro", uri=True)
    E = pd.read_sql("SELECT work_id AS rid, referenced_work_id AS ref FROM work_references", con)
    E = E[E["rid"].isin(pos.index)]
    df = E.groupby("ref").size()
    E = E[E["ref"].map(df) >= min_df]
    fi = pd.Series(np.arange(E["ref"].nunique()), index=E["ref"].unique())
    rows = pos[E["rid"]].values; cols = fi[E["ref"]].values
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(V), len(fi)))
    S = (X.T @ V).astype(np.float32)                      # feature sums of citing-paper vectors
    C = np.zeros_like(V); nref = np.asarray(X.sum(1)).ravel()
    Xc = X.tocsr()
    step = 20000
    for a in range(0, len(V), step):
        sub = Xc[a:a + step].tocoo()
        p = sub.row + a
        Y = S[sub.col] - V[p]                             # leave-one-out
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
        if len(p):
            u, st = np.unique(p, return_index=True)      # CSR -> COO rows are sorted
            C[u] += np.add.reduceat(Y, st, axis=0)
    C /= np.maximum(nref, 1)[:, None]
    return C, nref


def hill(cnt, q):
    p = cnt[cnt > 0] / cnt.sum()
    if q == 0:
        return float(len(p))
    return float(np.exp(-(p * np.log(p)).sum()))


def chao_shen(cnt):
    c = cnt[cnt > 0].astype(float); n = c.sum()
    if n < 2:
        return 0.0
    f1 = (c == 1).sum(); C = 1 - f1 / n if f1 < n else 1 / n
    p = C * c / n; den = 1 - (1 - p) ** n
    return float(-(p * np.log(p) / np.where(den > 0, den, 1)).sum())


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.arccos(np.clip(a @ b / (na * nb + 1e-12), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b2025")
    ap.add_argument("--policy", choices=["P1", "P3"], default="P1")
    ap.add_argument("--clusters", type=int, default=60)
    a = ap.parse_args()
    out = db.OUT / STEP; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(42)
    S = np.load(db.OUT / f"S08_space/space_{a.tag}.npz")
    pos = pd.Series(np.arange(len(S["ids"])), index=S["ids"]); V = S["vectors"].astype(np.float32)
    min_df = int(S["min_df"])
    w = json.loads((db.OUT / "S02_cells/diagnostics.json").read_text())["window_chosen"]

    from sklearn.cluster import MiniBatchKMeans
    lab = MiniBatchKMeans(n_clusters=a.clusters, random_state=0, n_init=5, batch_size=4096).fit(V).labels_
    C, nref = ref_coordinates(pos, V, min_df)
    print(f"space {V.shape}, clusters, ref coordinates done ({time.time() - t0:.0f}s)", flush=True)

    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz",
                       usecols=["professor_uid", "work_id", "year", "subject", "koad_action"])
    if a.policy == "P3":
        rows = rows[rows["koad_action"] != "review"]
    prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv").set_index("professor_uid")
    wk = pd.read_csv(db.OUT / "S08_space/works_kept.csv.gz", usecols=["work_id", "record_id"])
    n0 = len(rows)
    rows = rows[rows["professor_uid"].isin(prof.index) & (rows["year"] <= 2025)]
    n_curve = len(rows)
    rows = rows.merge(wk, on="work_id", how="left")
    no_coord = rows[rows["record_id"].isna()]
    rows = rows[rows["record_id"].notna()].copy()
    dup = rows.duplicated(["professor_uid", "record_id"])
    rows = rows[~dup]
    rows["i"] = pos[rows["record_id"]].values
    rows["win"] = (rows["year"] - rows["professor_uid"].map(prof["appointment_year"]).astype(int)) // w

    recs, dropped = [], []
    for (uid, subj), g in rows.groupby(["professor_uid", "subject"], sort=False):
        prev = None
        for win, h in g.groupby("win", sort=True):
            idx = h["i"].values; n = len(idx)
            if n < 2:
                dropped.append((uid, win, n, "n_lt_2"))
                continue
            X = V[idx]; m = X.mean(0); R2 = float(m @ m); R = np.sqrt(R2)
            c = m / (R + 1e-12)
            q_hat = 2 * n / (n - 1) * (1 - R2)
            r2c = (n * R2 - 1) / (n - 1)
            # half-split noise for this cell
            th_half2 = np.nan
            if n >= 4:
                hs = []
                for _ in range(K_SPLIT):
                    p_ = rng.permutation(n); k = n // 2
                    hs.append(angle(X[p_[:k]].mean(0), X[p_[k:]].mean(0)) ** 2)
                th_half2 = float(np.mean(hs))
            # integration ratio
            Cp = C[idx]; cbar = Cp.mean(0)
            within = float(np.mean(1 - (Cp * Cp).sum(1))); total = float(1 - cbar @ cbar)
            between = total - within
            cnt = np.bincount(lab[idx], minlength=a.clusters)
            top = int(cnt.argmax())
            r = {"professor_uid": uid, "subject": subj, "win": int(win), "et_start": int(win) * w, "n_papers": n,
                 "q_hat": q_hat, "R": R, "r2_corr": r2c, "d_corr": 1 - np.sqrt(max(0.0, r2c)),
                 "radius": float(np.mean(np.linalg.norm(X - c, axis=1))),
                 "integ_within": within, "integ_between": between, "integ": within / total if total > 1e-9 else np.nan,
                 "mean_refs": float(nref[idx].mean()),
                 "hill0": hill(cnt, 0), "hill1": hill(cnt, 1), "entropy_chao": chao_shen(cnt),
                 "n_clusters": int((cnt > 0).sum()), "top_cluster": top,
                 "review_share": float((h["koad_action"] == "review").mean()),
                 "gap": np.nan, "theta": np.nan, "theta0": np.nan, "theta_corr": np.nan,
                 "persistence": np.nan, "cluster_jump": np.nan, "shift2_u": np.nan}
            if prev is not None:
                r["gap"] = int(win) - prev["win"]
                th = angle(c, prev["c"]); r["theta"] = th
                if not np.isnan(th_half2) and not np.isnan(prev["half2"]):
                    t0_2 = th_half2 / 4 + prev["half2"] / 4
                    r["theta0"] = np.sqrt(t0_2); r["theta_corr"] = np.sqrt(max(0.0, th ** 2 - t0_2))
                if prev.get("shift") is not None:
                    v1, v2 = c - prev["c"], prev["shift"]
                    d = np.linalg.norm(v1) * np.linalg.norm(v2)
                    r["persistence"] = float(v1 @ v2 / d) if d > 1e-9 else np.nan
                r["cluster_jump"] = int(top != prev["top"])
                r["shift2_u"] = r2c + prev["r2c"] - 2 * float(m @ prev["m"])
            recs.append(r)
            prev = {"win": int(win), "c": c, "half2": th_half2, "top": top, "m": m, "r2c": r2c,
                    "shift": (c - prev["c"]) if prev is not None else None}
    P = pd.DataFrame(recs)
    P["appointment_year"] = P["professor_uid"].map(prof["appointment_year"]).astype(int)
    tag = f"{a.tag}_{a.policy}"
    P.to_csv(out / f"panel_{tag}.csv.gz", index=False)
    D = pd.DataFrame(dropped, columns=["professor_uid", "win", "n", "reason"])
    D.to_csv(out / f"dropped_cells_{tag}.csv.gz", index=False)
    no_coord.groupby("year").size().rename("rows").to_csv(out / f"rows_without_coordinate_{tag}.csv")
    L = [f"# {STEP} ({tag})", "",
         f"- rows S01 ({a.policy}) {n0:,}; curve rows (appointment known, <= 2025) {n_curve:,}; "
         f"without space coordinate {len(no_coord):,} ({len(no_coord) / n_curve:.1%}); same paper twice for a professor {int(dup.sum())}",
         f"- window {w} y; cells kept {len(P):,} (professors {P.professor_uid.nunique():,}); cells dropped n<2 {len(D):,}",
         f"- theta defined {P.theta.notna().sum():,}; theta_corr (n>=4 both) {P.theta_corr.notna().sum():,}; "
         f"share theta_corr == 0: {(P.theta_corr == 0).mean():.1%}",
         f"- medians: q_hat {P.q_hat.median():.3f}, r2_corr {P.r2_corr.median():.3f}, theta {P.theta.median():.3f}, "
         f"theta0 {P.theta0.median():.3f}, theta_corr {P.theta_corr.median():.3f}, integ {P.integ.median():.3f}, "
         f"shift2_u {P.shift2_u.median():.3f}",
         f"- theta_corr at the floor among defined: {(P.theta_corr.dropna() == 0).mean():.1%}",
         f"- clusters {a.clusters} (MiniBatchKMeans seed 0); {time.time() - t0:.0f}s"]
    (out / f"changes_{tag}.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
