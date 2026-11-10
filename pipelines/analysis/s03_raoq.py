#!/usr/bin/env python3
"""S03 Rao's Q-hat per professor x event-time window, from OpenAlex classification only.

Coordinates (handoff §3, x_p on the unit sphere up to scale):
  one-hot topic:   x_p = e_topic                        ||x_p - x_q||^2 = 2 * 1[t_p != t_q]
  tree distance:   x_p = (e_topic, e_subfield, e_field, e_domain) / sqrt(2)
                   ||x_p - x_q||^2 = number of hierarchy levels that differ = tree distance (0..4)
U-statistic  Q-hat = 1/(n(n-1)) sum_{p != q} ||x_p - x_q||^2,  unbiased for every n >= 2.
For one-hot coordinates at level L this is 2 * G_L with G_L = 1 - sum_k c_k (c_k - 1) / (n (n - 1))
(unbiased Gini-Simpson), so we report
  q_topic = G_topic                          in [0, 1]
  q_tree  = (G_topic + G_subfield + G_field + G_domain) / 4   in [0, 1]  (main index)
Plug-in comparators (biased, for the n-dependence check only):
  gs_plugin_topic = 1 - sum p_k^2,  shannon_plugin_topic = -sum p_k ln p_k

KOAD review policies (handoff §4: run all three):
  P1 decision  : KOAD namesakes and review rows without a career anchor excluded = main
  P2 review_keep: otherwise eligible review exclusions re-admitted within the S02 roster
  P3 review_drop: every KOAD review row dropped (only KOAD keep rows)
  flag column  : share of each cell's papers that are KOAD review rows (under P1)

Cells with n < 2 are dropped (saved with reason).
Outputs: S03_raoq/cells_w{w}.csv.gz (kept cells, all policies), dropped_w{w}.csv.gz, changes.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

STEP = "S03_raoq"
LAST_FULL_YEAR = 2025
LEVELS = ["topic_id", "subfield_id", "field_id", "domain_id"]


def cell_table(d2):
    """Vectorised per-cell statistics. d2 has columns professor_uid, subject, win, LEVELS, koad_action."""
    key = ["professor_uid", "subject", "win"]
    base = d2.groupby(key).agg(n=("topic_id", "size"),
                               review_share=("koad_action", lambda s: float((s == "review").mean())))
    n = base["n"].astype(float)
    G = {}
    for L in LEVELS:
        c = d2.groupby(key + [L]).size().astype(float)
        s_cc = (c * (c - 1)).groupby(level=key).sum()
        G[L] = 1.0 - s_cc.reindex(base.index) / (n * (n - 1))
        if L == "topic_id":
            p = c / c.groupby(level=key).transform("sum")
            base["gs_plugin_topic"] = 1 - (p ** 2).groupby(level=key).sum()
            base["shannon_plugin_topic"] = (-(p * np.log(p))).groupby(level=key).sum()
            base["n_topics"] = c.groupby(level=key).size()
    base["q_topic"] = G["topic_id"]
    base["q_subfield"] = G["subfield_id"]
    base["q_field"] = G["field_id"]
    base["q_tree"] = sum(G.values()) / 4
    return base.reset_index()


def load_policies():
    dt = {L: str for L in LEVELS}
    kept = pd.read_csv(db.OUT / "S01_select/kept.csv.gz", dtype=dt)
    drop = pd.read_csv(db.OUT / "S01_select/dropped.csv.gz", dtype=dt)
    prof = pd.read_csv(db.OUT / "S02_cells/professors_kept.csv")
    review_exclusions = {"koad_namesake", "trajectory_unanchored_review"}
    back = drop[drop["reason"].isin(review_exclusions) & (drop["koad_action"] == "review")
                & drop["work_type"].isin(db.JOURNAL_TYPES) & drop["year"].notna()
                & (drop["year"] <= db.MAX_YEAR) & drop["topic_id"].notna()]
    back = back.merge(prof[["professor_uid", "phd_year"]], on="professor_uid", how="left")
    back = back[~(back["phd_year"].notna() & (back["year"] < back["phd_year"] - 5))].drop(columns=["phd_year", "reason"])
    pol = {"P1_decision": kept,
           "P2_review_keep": pd.concat([kept, back], ignore_index=True),
           "P3_review_drop": kept[kept["koad_action"] != "review"]}
    appt = prof.set_index("professor_uid")["appointment_year"]
    for k, df in pol.items():
        df = df[df["professor_uid"].isin(appt.index) & (df["year"] <= LAST_FULL_YEAR)].copy()
        df["event_time"] = df["year"] - df["professor_uid"].map(appt).astype(int)
        pol[k] = df
    return pol, len(back)


def main():
    out = db.OUT / STEP
    out.mkdir(parents=True, exist_ok=True)
    pol, n_back = load_policies()
    L = [f"# {STEP}", "", f"- P2 re-admits {n_back:,} KOAD review exclusions (before appointment/2026 filters)"]
    for w in (1, 2, 3):
        kept_all, drop_all = [], []
        for name, df in pol.items():
            df = df.assign(win=np.floor_divide(df["event_time"], w))
            sizes = df.groupby(["professor_uid", "win"]).size()
            ok = sizes[sizes >= 2].index
            idx = df.set_index(["professor_uid", "win"]).index
            d2 = df[idx.isin(ok)]
            st = cell_table(d2)
            st["policy"] = name
            kept_all.append(st)
            dd = sizes[sizes < 2].rename("n").reset_index()
            dd["policy"] = name; dd["reason"] = "n_lt_2"
            drop_all.append(dd)
            L.append(f"- w={w} {name}: rows {len(df):,}; cells kept {len(st):,} (rows {int(st['n'].sum()):,}); "
                     f"cells dropped n<2 {len(dd):,}; professors {st['professor_uid'].nunique():,}")
        pd.concat(kept_all).to_csv(out / f"cells_w{w}.csv.gz", index=False)
        pd.concat(drop_all).to_csv(out / f"dropped_w{w}.csv.gz", index=False)
    (out / "changes.md").write_text("\n".join(L) + "\n")
    with open(db.OUT / "STEP_LOG.md", "a") as fh:
        fh.write("\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
