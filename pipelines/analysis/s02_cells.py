#!/usr/bin/env python3
"""S02 cells: attach first appointment year, build event_time, and report the pre-plot diagnostics.

event_time = publication_year - professors.appointment_year (first appointment, KRI roster).
Professors are dropped from the curve when appointment_year is missing, or flagged when
appointment_year < phd_year (kept, flag column) — both reported.

Diagnostics (all on S01 kept rows, main KOAD policy):
  - publication_year distribution
  - papers per professor (all 3,937 roster professors; 0 for those with no kept paper)
  - retained / excluded counts at row and professor level
  - share of professor-years with n < 2:
      (a) among active cells (n >= 1),
      (b) among span cells (first..last publication year, zeros included),
    for windows w = 1, 2, 3 years aligned on event_time (window = floor(event_time / w)),
    restricted to publication_year <= 2025 (2026 is a partial year).

Window rule (fixed before looking at the curve): use the smallest w whose active-cell share
with n < 2 is <= 20 %; if none, use w = 3.

Outputs: S02_cells/cells_w{1,2,3}.csv.gz, professors_kept.csv, professors_dropped.csv,
         diagnostics.md, diagnostics.json, fig_diagnostics.pdf/png
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import style  # noqa: E402

STEP = "S02_cells"
LAST_FULL_YEAR = 2025
THRESH = 0.20


def main():
    out = db.OUT / STEP
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", dtype={"topic_id": str, "subfield_id": str,
                                                                  "field_id": str, "domain_id": str})
    con = db.ro(db.WEB)
    prof = pd.read_sql("SELECT professor_uid, subject, appointment_year, phd_year FROM professors", con)
    n_by_prof = rows.groupby("professor_uid").size()
    prof["n_papers"] = prof["professor_uid"].map(n_by_prof).fillna(0).astype(int)
    prof["appt_before_phd"] = (prof["appointment_year"] < prof["phd_year"]).fillna(False)

    drop_reason = np.where(prof["n_papers"] == 0, "no_kept_paper",
                  np.where(prof["appointment_year"].isna(), "no_appointment_year", ""))
    prof["reason"] = drop_reason
    prof_kept = prof[prof["reason"] == ""].drop(columns="reason")
    prof_drop = prof[prof["reason"] != ""]
    prof_kept.to_csv(out / "professors_kept.csv", index=False)
    prof_drop.to_csv(out / "professors_dropped.csv", index=False)

    r = rows.merge(prof_kept[["professor_uid", "appointment_year"]], on="professor_uid", how="inner")
    rows_no_appt = len(rows) - len(r)
    r = r[r["year"] <= LAST_FULL_YEAR].copy()
    rows_2026 = len(rows) - rows_no_appt - len(r)
    r["event_time"] = r["year"] - r["appointment_year"].astype(int)

    diag = {"rows_S01_kept": int(len(rows)), "rows_dropped_no_appointment": int(rows_no_appt),
            "rows_dropped_year_2026_partial": int(rows_2026), "rows_curve": int(len(r)),
            "professors_roster": int(len(prof)),
            "professors_dropped": prof_drop["reason"].value_counts().to_dict(),
            "professors_curve": int(r["professor_uid"].nunique()),
            "professors_appt_before_phd_flagged": int(prof_kept["appt_before_phd"].sum())}

    ppp = prof["n_papers"]
    q = [0, .1, .25, .5, .75, .9, .99, 1]
    diag["papers_per_professor_all_roster"] = {f"q{int(x*100)}": float(ppp.quantile(x)) for x in q}
    diag["papers_per_professor_all_roster"]["mean"] = float(ppp.mean())
    pk = prof_kept["n_papers"]
    diag["papers_per_professor_curve"] = {f"q{int(x*100)}": float(pk.quantile(x)) for x in q}
    diag["papers_per_professor_curve"]["mean"] = float(pk.mean())
    diag["papers_per_professor_bins"] = {
        k: int(v) for k, v in pd.cut(ppp, [-1, 0, 4, 9, 19, 49, 99, 199, 10**6],
                                    labels=["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200+"]
                                    ).value_counts().sort_index().items()}
    diag["by_subject"] = {s: {"professors": int(g["professor_uid"].nunique()), "rows": int(len(g))}
                          for s, g in r.groupby("subject")}

    # professor-year cells
    win = {}
    for w in (1, 2, 3):
        rr = r.assign(win=np.floor_divide(r["event_time"], w))
        cells = rr.groupby(["professor_uid", "subject", "win"]).size().rename("n").reset_index()
        cells.to_csv(out / f"cells_w{w}.csv.gz", index=False)
        active_lt2 = float((cells["n"] < 2).mean())
        # span cells: first..last active window per professor (zeros included)
        span = cells.groupby("professor_uid")["win"].agg(["min", "max"])
        n_span = int((span["max"] - span["min"] + 1).sum())
        span_lt2 = (n_span - int((cells["n"] >= 2).sum())) / n_span
        by_subj = {s: float((g["n"] < 2).mean()) for s, g in cells.groupby("subject")}
        lost_rows = int(cells.loc[cells["n"] < 2, "n"].sum())
        win[w] = {"active_cells": int(len(cells)), "active_share_n_lt_2": active_lt2,
                  "span_cells": n_span, "span_share_n_lt_2": span_lt2,
                  "rows_lost_in_n1_cells": lost_rows, "active_share_n_lt_2_by_subject": by_subj,
                  "median_n_active": float(cells["n"].median())}
    diag["windows"] = win
    chosen = next((w for w in (1, 2, 3) if win[w]["active_share_n_lt_2"] <= THRESH), 3)
    diag["window_rule"] = f"smallest w with active-cell share n<2 <= {THRESH:.0%}"
    diag["window_chosen"] = chosen
    diag["year_distribution_kept"] = {int(k): int(v) for k, v in rows["year"].value_counts().sort_index().items()}
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1, ensure_ascii=False))

    L = [f"# {STEP}", "",
         f"- rows S01 kept {len(rows):,}; dropped no appointment_year {rows_no_appt:,}; "
         f"dropped 2026 (partial) {rows_2026:,}; curve rows {len(r):,}",
         f"- professors roster {len(prof):,}; dropped {diag['professors_dropped']}; curve {diag['professors_curve']:,}; "
         f"flagged appt<phd {diag['professors_appt_before_phd_flagged']}",
         f"- papers/professor (all roster): " + ", ".join(f"{k} {v:.0f}" for k, v in diag['papers_per_professor_all_roster'].items()),
         f"- papers/professor bins: {diag['papers_per_professor_bins']}", "",
         "| window | active cells | share n<2 (active) | share n<2 (span, zeros incl.) | median n | rows in n=1 cells |",
         "|---|---|---|---|---|---|"]
    for w in (1, 2, 3):
        d = win[w]
        L.append(f"| {w} y | {d['active_cells']:,} | {d['active_share_n_lt_2']:.1%} | {d['span_share_n_lt_2']:.1%} "
                 f"| {d['median_n_active']:.0f} | {d['rows_lost_in_n1_cells']:,} |")
    L += ["", "by subject (active share n<2): " + "; ".join(
        f"w={w}: " + ", ".join(f"{s} {v:.1%}" for s, v in win[w]['active_share_n_lt_2_by_subject'].items())
        for w in (1, 2, 3)),
          "", f"window rule: {diag['window_rule']} -> chosen w = {chosen}"]
    (out / "diagnostics.md").write_text("\n".join(L) + "\n")
    with open(db.OUT / "STEP_LOG.md", "a") as fh:
        fh.write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))
    figure(rows, prof, out, win)


def figure(rows, prof, out, win):
    plt = style.plt
    fig, ax = plt.subplots(1, 3, figsize=(style.W, style.W * 0.36))
    yc = rows["year"].value_counts().sort_index()
    ax[0].bar(yc.index, yc.values, width=0.9, color=style.C[0], edgecolor="none")
    ax[0].bar([2026], [yc.get(2026, 0)], width=0.9, color=style.GREY, edgecolor="none")
    ax[0].set_xlabel("Publication year"); ax[0].set_ylabel("Paper rows")
    ax[0].text(0.03, 0.95, "2026 partial (to Aug 23)", transform=ax[0].transAxes, va="top", color=style.GREY)
    n = prof["n_papers"]
    bins = np.unique(np.round(np.logspace(0, np.log10(n.max() + 1), 30)))
    ax[1].hist(n[n > 0], bins=bins, color=style.C[0], edgecolor="none")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Papers per professor"); ax[1].set_ylabel("Professors")
    ax[1].axvline(n[n > 0].median(), color="k", lw=0.6, ls="--")
    ws = [1, 2, 3]
    ax[2].bar([w - 0.18 for w in ws], [win[w]["active_share_n_lt_2"] for w in ws], 0.36,
              color=style.C[0], label="active cells")
    ax[2].bar([w + 0.18 for w in ws], [win[w]["span_share_n_lt_2"] for w in ws], 0.36,
              color=style.C[1], label="span cells (zeros incl.)")
    ax[2].axhline(THRESH, color="k", lw=0.6, ls="--")
    ax[2].set_xticks(ws, [f"{w} y" for w in ws]); ax[2].set_xlabel("Window")
    ax[2].set_ylabel("Share of cells with n < 2"); ax[2].legend(frameon=False, loc="upper right")
    for a, l in zip(ax, "abc"):
        a.text(-0.22, 1.02, l, transform=a.transAxes, fontweight="bold", fontsize=style.FS + 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_diagnostics.{ext}", dpi=300)


if __name__ == "__main__":
    main()
