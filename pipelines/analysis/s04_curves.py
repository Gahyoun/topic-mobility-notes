#!/usr/bin/env python3
"""S04 preliminary Q0 curve: career event time vs Rao's Q-hat (classification coordinates).

Main spec: policy P1_decision, window w (from S02 diagnostics), index q_tree.
  (a) raw mean of q_tree per subject x event-time window, 95% CI by professor bootstrap
  (b) person fixed-effect profile: q_ic = alpha_i + beta_w + e_ic (per subject, alternating projections),
      beta normalised to the window containing event_time 0; professor bootstrap CI
  (c) n-dependence check: plug-in Gini-Simpson vs unbiased q_topic as a function of n
  (d) field-wide q_tree by calendar year (random paper pairs within subject; period reference —
      with person FE, event_time slope equals calendar-time slope, so period drift is not separated)
Sensitivity: KOAD policy P1/P2/P3, window w and w+1, q_topic instead of q_tree.
Summary: within-person linear slope of q_tree per decade of event time in [0, 30), per subject.

Windows shown only when at least MIN_PROF professors contribute. Outputs: S04_curves/*.csv, *.md, figures.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import style  # noqa: E402

STEP = "S04_curves"
MIN_PROF = 30
B = 200
ET_MIN, ET_MAX = -6, 36
SUBJECTS = ["physics", "chemistry", "biology", "mathematics"]
rng = np.random.default_rng(20261104)


def fe_profile(pid, win, y, n_win, iters=500, tol=1e-10):
    """Two-way within estimator by alternating projections. Returns beta per window (nan if empty)."""
    P = pid.max() + 1
    cp = np.bincount(pid, minlength=P).astype(float)
    cw = np.bincount(win, minlength=n_win).astype(float)
    a = np.zeros(P); b = np.zeros(n_win)
    for _ in range(iters):
        a_new = np.bincount(pid, y - b[win], minlength=P) / np.maximum(cp, 1)
        b_new = np.bincount(win, y - a_new[pid], minlength=n_win) / np.maximum(cw, 1)
        done = np.max(np.abs(b_new - b)) < tol
        a, b = a_new, b_new
        if done:
            break
    b[cw == 0] = np.nan
    return b


def fe_slope(pid, x, y):
    """Within-person OLS slope of y on x."""
    P = pid.max() + 1
    cp = np.bincount(pid, minlength=P).astype(float)
    cp = np.maximum(cp, 1)
    xd = x - (np.bincount(pid, x, minlength=P) / cp)[pid]
    yd = y - (np.bincount(pid, y, minlength=P) / cp)[pid]
    return float((xd * yd).sum() / (xd * xd).sum())


def subject_curves(c, w, col):
    """c: cells of one subject. Returns DataFrame per window with raw mean, FE beta, CIs."""
    c = c[(c["win"] * w >= ET_MIN) & (c["win"] * w < ET_MAX)].copy()
    wins = np.sort(c["win"].unique())
    wix = {v: i for i, v in enumerate(wins)}
    c["wi"] = c["win"].map(wix)
    uids = c["professor_uid"].unique()
    uix = {u: i for i, u in enumerate(uids)}
    c["pi"] = c["professor_uid"].map(uix)
    ref = wix.get(0)
    PI = c["pi"].values; WI = c["wi"].values; Y = c[col].values; WIN = c["win"].values
    order = np.argsort(PI, kind="stable")
    starts = np.searchsorted(PI[order], np.arange(len(uids)))
    ends = np.searchsorted(PI[order], np.arange(len(uids)), side="right")
    nW = len(wins)

    def stats(pid, wi, y, win):
        cnt = np.bincount(wi, minlength=nW)
        raw = np.bincount(wi, y, minlength=nW) / np.maximum(cnt, 1)
        raw[cnt == 0] = np.nan
        b = fe_profile(pid, wi, y, nW)
        b = b - b[ref]
        m = (win * w >= 0) & (win * w < 30)
        et = win * w + (w - 1) / 2
        s = fe_slope(pid[m], et[m], y[m]) * 10
        return raw, b, s

    raw, beta, slope = stats(PI, WI, Y, WIN)
    BR, BB, BS = [], [], []
    for _ in range(B):
        pick = rng.integers(0, len(uids), len(uids))
        idx = np.concatenate([order[starts[j]:ends[j]] for j in pick])
        pid = np.repeat(np.arange(len(pick)), ends[pick] - starts[pick])
        r_, b_, s_ = stats(pid, WI[idx], Y[idx], WIN[idx])
        BR.append(r_); BB.append(b_); BS.append(s_)
    BR, BB = np.array(BR), np.array(BB)
    nprof = c.groupby("wi")["professor_uid"].nunique().reindex(range(len(wins))).fillna(0).values
    df = pd.DataFrame({"win": wins, "et_mid": wins * w + (w - 1) / 2, "n_prof": nprof,
                       "n_cells": c.groupby("wi").size().reindex(range(len(wins))).fillna(0).values,
                       "raw": raw, "raw_lo": np.nanpercentile(BR, 2.5, 0), "raw_hi": np.nanpercentile(BR, 97.5, 0),
                       "fe": beta, "fe_lo": np.nanpercentile(BB, 2.5, 0), "fe_hi": np.nanpercentile(BB, 97.5, 0)})
    df.loc[df["n_prof"] < MIN_PROF, ["raw", "raw_lo", "raw_hi", "fe", "fe_lo", "fe_hi"]] = np.nan
    return df, (slope, float(np.percentile(BS, 2.5)), float(np.percentile(BS, 97.5)), len(uids))


def field_period(w_rows):
    """Field-wide q_tree by calendar year: unbiased, from level counts over all papers of a subject-year."""
    out = []
    for (s, y), g in w_rows.groupby(["subject", "year"]):
        n = len(g)
        if n < 50:
            continue
        G = []
        for L in ["topic_id", "subfield_id", "field_id", "domain_id"]:
            c = g[L].value_counts().values.astype(float)
            G.append(1 - (c * (c - 1)).sum() / (n * (n - 1)))
        out.append({"subject": s, "year": y, "n": n, "q_tree": np.mean(G), "q_topic": G[0]})
    return pd.DataFrame(out)


def main():
    out = db.OUT / STEP
    out.mkdir(parents=True, exist_ok=True)
    w = json.loads((db.OUT / "S02_cells/diagnostics.json").read_text())["window_chosen"]
    specs = [("main", "P1_decision", w, "q_tree"), ("P2_review_keep", "P2_review_keep", w, "q_tree"),
             ("P3_review_drop", "P3_review_drop", w, "q_tree"), (f"w{w + 1}", "P1_decision", w + 1, "q_tree"),
             ("q_topic", "P1_decision", w, "q_topic")]
    cells = {ww: pd.read_csv(db.OUT / f"S03_raoq/cells_w{ww}.csv.gz") for ww in {w, w + 1}}
    curves, slopes = [], []
    for name, pol, ww, col in specs:
        c = cells[ww][cells[ww]["policy"] == pol]
        for s in SUBJECTS:
            df, sl = subject_curves(c[c["subject"] == s], ww, col)
            df["spec"] = name; df["subject"] = s
            curves.append(df)
            slopes.append({"spec": name, "subject": s, "slope_per_decade": sl[0], "lo": sl[1], "hi": sl[2],
                           "professors": sl[3]})
            print(f"{name:15s} {s:12s} slope/decade {sl[0]:+.4f} [{sl[1]:+.4f}, {sl[2]:+.4f}] n_prof {sl[3]}", flush=True)
    curves = pd.concat(curves); slopes = pd.DataFrame(slopes)
    curves.to_csv(out / "curves.csv", index=False); slopes.to_csv(out / "slopes.csv", index=False)

    # n-dependence check (main policy, window w, pooled)
    c = cells[w][cells[w]["policy"] == "P1_decision"]
    nb = pd.cut(c["n"], [1, 2, 3, 4, 5, 7, 10, 15, 25, 1000], labels=["2", "3", "4", "5", "6-7", "8-10", "11-15", "16-25", "26+"])
    ndep = c.groupby(nb, observed=True)[["q_topic", "gs_plugin_topic", "q_tree", "shannon_plugin_topic"]].mean()
    ndep["cells"] = c.groupby(nb, observed=True).size()
    ndep.to_csv(out / "n_dependence.csv")

    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", dtype=str)
    rows["year"] = rows["year"].astype(int)
    fp = field_period(rows[rows["year"].between(1990, 2025)])
    fp.to_csv(out / "field_period.csv", index=False)

    L = [f"# {STEP}", "", f"window w = {w} (from S02), MIN_PROF {MIN_PROF}, bootstrap B = {B} (professor resample)", "",
         "within-person slope of index per decade of event time, event_time in [0, 30):", "",
         "| spec | subject | slope / decade | 95% CI | professors |", "|---|---|---|---|---|"]
    L += [f"| {r.spec} | {r.subject} | {r.slope_per_decade:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | {r.professors:,} |"
          for r in slopes.itertuples()]
    L += ["", "n-dependence (pooled, main): mean index by cell size n", "", ndep.round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    with open(db.OUT / "STEP_LOG.md", "a") as fh:
        fh.write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))
    figures(curves, ndep, fp, w, out)


def figures(curves, ndep, fp, w, out):
    plt = style.plt
    # Figure 1: main curves
    fig, ax = plt.subplots(2, 2, figsize=(style.W, style.W * 0.62))
    m = curves[curves["spec"] == "main"]
    for s in SUBJECTS:
        d = m[m["subject"] == s]
        col = style.SUBJ[s]
        ax[0, 0].plot(d["et_mid"], d["raw"], color=col, lw=1, label=s.capitalize())
        ax[0, 0].fill_between(d["et_mid"], d["raw_lo"], d["raw_hi"], color=col, alpha=0.18, lw=0)
        ax[0, 1].plot(d["et_mid"], d["fe"], color=col, lw=1)
        ax[0, 1].fill_between(d["et_mid"], d["fe_lo"], d["fe_hi"], color=col, alpha=0.18, lw=0)
    for a in ax[0]:
        a.axvspan(-1, 1, color=style.GREY, alpha=0.12, lw=0)
        a.set_xlabel("Career year (publication year − first appointment)")
    ax[0, 0].set_ylabel(r"Rao's $\hat{Q}$ (tree distance), mean")
    ax[0, 1].set_ylabel(r"$\hat{Q}$ person-FE profile, rel. to year 0")
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 0].legend(frameon=False, ncol=2, loc="lower right")
    x = np.arange(len(ndep))
    ax[1, 0].plot(x, ndep["q_topic"], "o-", color=style.C[0], ms=3, lw=1, label=r"unbiased $\hat{Q}$ (topic one-hot)")
    ax[1, 0].plot(x, ndep["gs_plugin_topic"], "s--", color=style.C[5], ms=3, lw=1, label="plug-in Gini–Simpson")
    ax[1, 0].set_xticks(x, ndep.index.astype(str)); ax[1, 0].set_xlabel(f"Papers per {w}-year cell, n")
    ax[1, 0].set_ylabel("Mean index"); ax[1, 0].legend(frameon=False, loc="lower right")
    for s in SUBJECTS:
        d = fp[fp["subject"] == s]
        ax[1, 1].plot(d["year"], d["q_tree"], color=style.SUBJ[s], lw=1)
    ax[1, 1].set_xlabel("Publication year"); ax[1, 1].set_ylabel(r"Field-wide $\hat{Q}$ (subject-year)")
    for a, l in zip(ax.flat, "abcd"):
        a.text(-0.16, 1.02, l, transform=a.transAxes, fontweight="bold", fontsize=style.FS + 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_q0_curve.{ext}", dpi=300)

    # Figure 2: sensitivity of the FE profile
    specs = ["main", "P2_review_keep", "P3_review_drop", f"w{w + 1}", "q_topic"]
    lab = {"main": f"P1 decision, w={w}, tree", "P2_review_keep": "P2 review→keep", "P3_review_drop": "P3 review→drop",
           f"w{w + 1}": f"window {w + 1} y", "q_topic": "topic one-hot"}
    ls = {"main": "-", "P2_review_keep": "--", "P3_review_drop": ":", f"w{w + 1}": "-.", "q_topic": (0, (1, 1, 3, 1))}
    fig, ax = plt.subplots(1, 4, figsize=(style.W, style.W * 0.3), sharey=True)
    for a, s in zip(ax, SUBJECTS):
        for sp in specs:
            d = curves[(curves["spec"] == sp) & (curves["subject"] == s)]
            a.plot(d["et_mid"], d["fe"], ls=ls[sp], color=style.SUBJ[s] if sp == "main" else "k",
                   lw=1 if sp == "main" else 0.7, label=lab[sp])
        a.axhline(0, color=style.GREY, lw=0.4); a.set_title(s.capitalize())
        a.set_xlabel("Career year")
    ax[0].set_ylabel(r"$\hat{Q}$ person-FE profile")
    ax[0].legend(frameon=False, fontsize=style.FS - 2, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_q0_sensitivity.{ext}", dpi=300)


if __name__ == "__main__":
    main()
