#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# BD-rate
# ============================================================
def bd_rate(
    rate_anchor: List[float],
    psnr_anchor: List[float],
    rate_test: List[float],
    psnr_test: List[float],
) -> float:
    r1 = np.array(rate_anchor, dtype=np.float64)
    p1 = np.array(psnr_anchor, dtype=np.float64)
    r2 = np.array(rate_test, dtype=np.float64)
    p2 = np.array(psnr_test, dtype=np.float64)

    m1 = np.isfinite(r1) & np.isfinite(p1) & (r1 > 0)
    m2 = np.isfinite(r2) & np.isfinite(p2) & (r2 > 0)
    r1, p1 = r1[m1], p1[m1]
    r2, p2 = r2[m2], p2[m2]

    if len(r1) < 2 or len(r2) < 2:
        return float("nan")

    i1 = np.argsort(p1)
    i2 = np.argsort(p2)
    p1, r1 = p1[i1], r1[i1]
    p2, r2 = p2[i2], r2[i2]

    p_min = max(p1.min(), p2.min())
    p_max = min(p1.max(), p2.max())
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return float("nan")

    lr1 = np.log(np.maximum(r1, 1e-12))
    lr2 = np.log(np.maximum(r2, 1e-12))

    deg1 = min(3, len(p1) - 1)
    deg2 = min(3, len(p2) - 1)
    if deg1 < 1 or deg2 < 1:
        return float("nan")

    c1 = np.polyfit(p1, lr1, deg1)
    c2 = np.polyfit(p2, lr2, deg2)

    ic1 = np.polyint(c1)
    ic2 = np.polyint(c2)

    int1 = np.polyval(ic1, p_max) - np.polyval(ic1, p_min)
    int2 = np.polyval(ic2, p_max) - np.polyval(ic2, p_min)

    avg_diff = (int2 - int1) / (p_max - p_min)
    return (math.exp(avg_diff) - 1.0) * 100.0


# ============================================================
# Helpers
# ============================================================
def mean_finite(xs: List[float]) -> float:
    ys = [float(x) for x in xs if np.isfinite(x)]
    if not ys:
        return float("nan")
    return float(np.mean(ys))


def parse_qps(s: str) -> List[int]:
    qps = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(qps) < 2:
        raise ValueError("Need at least 2 QPs")
    return qps


def compute_bdr_for_group(
    df_group: pd.DataFrame,
    qps: List[int],
    plane: str,
) -> float:
    """
    anchor = rec
    test   = rec_post
    """
    col_anchor = f"psnr{plane}_rec"
    col_test = f"psnr{plane}_rec_post"

    sub = df_group[df_group["qp"].isin(qps)].copy()
    if sub.empty:
        return float("nan")

    # enforce requested qp order only
    sub = sub.sort_values("qp")
    sub = sub.drop_duplicates(subset=["qp"], keep="first")

    # require all requested qp points if possible
    if len(sub) < len(qps):
        return float("nan")

    rates = sub["kbps"].astype(float).tolist()
    psnr_anchor = sub[col_anchor].astype(float).tolist()
    psnr_test = sub[col_test].astype(float).tolist()

    return bd_rate(
        rate_anchor=rates,
        psnr_anchor=psnr_anchor,
        rate_test=rates,
        psnr_test=psnr_test,
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="input psnr summary csv")
    ap.add_argument("--qps", type=str, default="22,27,32,37", help="comma-separated qp list")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = csv_path.parent / "bdrate.csv"
    qps = parse_qps(args.qps)

    df = pd.read_csv(csv_path)

    required_cols = [
        "qp", "seq_cls", "seq_name", "kbps",
        "psnrY_rec", "psnrU_rec", "psnrV_rec",
        "psnrY_rec_post", "psnrU_rec_post", "psnrV_rec_post",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------
    # per-sequence
    # ------------------------------------------------------------
    seq_bdr_rows: List[Dict[str, object]] = []

    for (seq_cls, seq_name), g in df.groupby(["seq_cls", "seq_name"], sort=True):
        bdr_y = compute_bdr_for_group(g, qps, "Y")
        bdr_u = compute_bdr_for_group(g, qps, "U")
        bdr_v = compute_bdr_for_group(g, qps, "V")

        row = {
            "level": "sequence",
            "seq_cls": seq_cls,
            "seq_name": seq_name,
            "n_qp": int(g[g["qp"].isin(qps)]["qp"].nunique()),
            "qps": ",".join(str(x) for x in qps),
            "bdrate_Y": bdr_y,
            "bdrate_U": bdr_u,
            "bdrate_V": bdr_v,
        }
        rows.append(row)
        seq_bdr_rows.append(row)

    seq_bdr_df = pd.DataFrame(seq_bdr_rows)

    # ------------------------------------------------------------
    # per-class mean over sequences
    # ------------------------------------------------------------
    for seq_cls, g in seq_bdr_df.groupby("seq_cls", sort=True):
        row = {
            "level": "class_mean",
            "seq_cls": seq_cls,
            "seq_name": "",
            "n_qp": len(qps),
            "qps": ",".join(str(x) for x in qps),
            "bdrate_Y": mean_finite(g["bdrate_Y"].tolist()),
            "bdrate_U": mean_finite(g["bdrate_U"].tolist()),
            "bdrate_V": mean_finite(g["bdrate_V"].tolist()),
        }
        rows.append(row)

    # ------------------------------------------------------------
    # overall mean over sequences
    # ------------------------------------------------------------
    row_all = {
        "level": "overall_mean",
        "seq_cls": "ALL",
        "seq_name": "",
        "n_qp": len(qps),
        "qps": ",".join(str(x) for x in qps),
        "bdrate_Y": mean_finite(seq_bdr_df["bdrate_Y"].tolist()),
        "bdrate_U": mean_finite(seq_bdr_df["bdrate_U"].tolist()),
        "bdrate_V": mean_finite(seq_bdr_df["bdrate_V"].tolist()),
    }
    rows.append(row_all)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
