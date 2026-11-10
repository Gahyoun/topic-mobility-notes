#!/usr/bin/env python3
"""S12 first regression (handoff §8.4): event time vs panel quantities with person fixed effects, by subject.

Per subject: y_ic = alpha_i + beta_window + e  (profile, rel. to the window with event_time 0) and the within-person
linear slope per decade on event_time in [0, 30); professor bootstrap CIs (functions from S04).
Outcomes: embedding q_hat (P1 main, P3 sensitivity), classification q_tree on the same cells (S03),
theta_corr, integ. Period reference: field-wide embedding Q-hat by calendar year (all papers of a subject-year).

  python3 -I analysis/s12_first_regression.py --tag b2025
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import style  # noqa: E402
from s04_curves import SUBJECTS, subject_curves  # noqa: E402

STEP = "S12_first_regression"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="b2025")
    a = ap.parse_args()
    out = db.OUT / STEP; out.mkdir(parents=True, exist_ok=True)
    w = json.loads((db.OUT / "S02_cells/diagnostics.json").read_text())["window_chosen"]
    P1 = pd.read_csv(db.OUT / f"S11_panel/panel_{a.tag}_P1.csv.gz")
    P3 = pd.read_csv(db.OUT / f"S11_panel/panel_{a.tag}_P3.csv.gz")
    c3 = pd.read_csv(db.OUT / f"S03_raoq/cells_w{w}.csv.gz")
    c3 = c3[c3.policy == "P1_decision"][["professor_uid", "win", "q_tree"]]
    P1c = P1.merge(c3, on=["professor_uid", "win"], how="inner")
    specs = [("q_hat", P1, "q_hat"), ("q_hat_P3", P3, "q_hat"), ("q_tree_same_cells", P1c, "q_tree"),
             ("q_hat_same_cells", P1c, "q_hat"), ("shift2_u", P1, "shift2_u"), ("theta_corr", P1, "theta_corr"), ("integ", P1, "integ")]
    curves, slopes = [], []
    for name, D, col in specs:
        D = D[D[col].notna()]
        for s in SUBJECTS:
            df, sl = subject_curves(D[D.subject == s], w, col)
            df["spec"] = name; df["subject"] = s; curves.append(df)
            slopes.append({"spec": name, "subject": s, "slope_per_decade": sl[0], "lo": sl[1], "hi": sl[2],
                           "professors": sl[3], "cells": int((D.subject == s).sum()),
                           "y_sd": float(D.loc[D.subject == s, col].std())})
    curves = pd.concat(curves); slopes = pd.DataFrame(slopes)
    slopes["slope_in_sd"] = slopes.slope_per_decade / slopes.y_sd
    curves.to_csv(out / f"curves_{a.tag}.csv", index=False); slopes.to_csv(out / f"slopes_{a.tag}.csv", index=False)

    # period reference: field-wide embedding Q-hat by calendar year
    S = np.load(db.OUT / f"S08_space/space_{a.tag}.npz")
    pos = pd.Series(np.arange(len(S["ids"])), index=S["ids"]); V = S["vectors"].astype(np.float64)
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", usecols=["work_id", "year", "subject"])
    wk = pd.read_csv(db.OUT / "S08_space/works_kept.csv.gz", usecols=["work_id", "record_id"])
    rows = rows.merge(wk, on="work_id").drop_duplicates(["subject", "record_id"])
    fp = []
    for (s, y), g in rows[rows.year.between(1990, 2025)].groupby(["subject", "year"]):
        n = len(g)
        if n < 50:
            continue
        m = V[pos[g.record_id].values].mean(0)
        fp.append({"subject": s, "year": y, "n": n, "q_hat": 2 * n / (n - 1) * (1 - m @ m)})
    fp = pd.DataFrame(fp); fp.to_csv(out / f"field_period_{a.tag}.csv", index=False)
    per = {s: np.polyfit(d.year[d.year >= 2000], d.q_hat[d.year >= 2000], 1)[0] * 10 for s, d in fp.groupby("subject")}

    L = [f"# {STEP} ({a.tag})", "", "within-person slope per decade, event_time in [0, 30), 95% professor bootstrap CI:", "",
         "| spec | subject | slope / decade | 95% CI | slope / SD | professors | cells |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r.spec} | {r.subject} | {r.slope_per_decade:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | {r.slope_in_sd:+.3f} | "
          f"{r.professors:,} | {r.cells:,} |" for r in slopes.itertuples()]
    L += ["", "field-wide embedding Q-hat trend per decade (calendar 2000–2025; period reference): "
          + ", ".join(f"{s} {v:+.4f}" for s, v in per.items())]
    (out / f"summary_{a.tag}.md").write_text("\n".join(L) + "\n")
    open(db.OUT / "STEP_LOG.md", "a").write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))
    figure(curves, fp, out, a.tag)


def figure(curves, fp, out, tag):
    plt = style.plt
    fig, ax = plt.subplots(2, 2, figsize=(style.W, style.W * 0.62))
    panels = [(ax[0, 0], "q_hat", r"Embedding $\hat{Q}$, person-FE"),
              (ax[0, 1], "shift2_u", r"Shift $\|\Delta\mu\|^2$ (U-stat.), FE"),
              (ax[1, 0], "integ", "Integration ratio $I$, FE")]
    for a_, spec, lab in panels:
        for s in SUBJECTS:
            d = curves[(curves.spec == spec) & (curves.subject == s)]
            a_.plot(d.et_mid, d.fe, color=style.SUBJ[s], lw=1, label=s.capitalize())
            a_.fill_between(d.et_mid, d.fe_lo, d.fe_hi, color=style.SUBJ[s], alpha=0.18, lw=0)
        a_.axhline(0, color="k", lw=0.5); a_.axvspan(-1, 1, color=style.GREY, alpha=0.12, lw=0)
        a_.set_ylabel(lab); a_.set_xlabel("Career year (publication year − first appointment)")
    ax[0, 0].legend(frameon=False, ncol=2, loc="lower right")
    for s in SUBJECTS:
        d = fp[fp.subject == s]
        ax[1, 1].plot(d.year, d.q_hat, color=style.SUBJ[s], lw=1)
    ax[1, 1].set_xlabel("Publication year"); ax[1, 1].set_ylabel(r"Field-wide embedding $\hat{Q}$")
    for a_, l in zip(ax.flat, "abcd"):
        a_.text(-0.16, 1.02, l, transform=a_.transAxes, fontweight="bold", fontsize=style.FS + 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_first_regression_{tag}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
