#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except ImportError as e:
    raise ImportError("scipy is required. Install with: pip install scipy") from e


# ============================================================
# Parse args helpers
# ============================================================
def parse_str_list(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty string list argument")
    return vals


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty int list argument")
    return vals


# ============================================================
# Column name helpers
# ============================================================
def col_delta_kbps(sigma: str, qp: int) -> str:
    return f"delta_kbps_{sigma}_qp{qp}"


def col_psnr(sigma: str, qp: int) -> str:
    return f"psnrY_blur_deblur_{sigma}_qp{qp}"


# ============================================================
# Models
# ============================================================
def logistic4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    4-parameter logistic:
        y = d + (a - d) / (1 + exp(-b * (x - c)))
    a: lower-ish asymptote
    d: upper-ish asymptote
    b: slope
    c: inflection center
    """
    z = np.clip(-b * (x - c), -60.0, 60.0)
    return d + (a - d) / (1.0 + np.exp(z))


def poly2(x: np.ndarray, p2: float, p1: float, p0: float) -> np.ndarray:
    return p2 * x * x + p1 * x + p0


# ============================================================
# Metrics
# ============================================================
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


# ============================================================
# Fitting
# ============================================================
def fit_logistic4(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      popt: [a, b, c, d]
      yhat
      r2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Robust-ish init
    a0 = float(np.percentile(y, 10))
    d0 = float(np.percentile(y, 90))
    c0 = float(np.median(x))
    x_span = max(float(np.max(x) - np.min(x)), 1e-6)
    b0 = 4.0 / x_span

    p0 = [a0, b0, c0, d0]

    # Loose bounds
    y_pad = max(1.0, 0.5 * float(np.std(y)) if len(y) > 1 else 1.0)
    lower = [float(np.min(y) - y_pad * 2.0), -100.0, float(np.min(x) - x_span), float(np.min(y) - y_pad * 2.0)]
    upper = [float(np.max(y) + y_pad * 2.0), 100.0, float(np.max(x) + x_span), float(np.max(y) + y_pad * 2.0)]

    popt, _ = curve_fit(
        logistic4,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000,
    )
    yhat = logistic4(x, *popt)
    r2 = r2_score(y, yhat)
    return popt, yhat, r2


def fit_poly2(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      coeffs: [p2, p1, p0]
      yhat
      r2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    coeffs = np.polyfit(x, y, deg=2)
    yhat = np.polyval(coeffs, x)
    r2 = r2_score(y, yhat)
    return coeffs, yhat, r2


# ============================================================
# Plot
# ============================================================
def plot_scatter_and_fit(
    out_png: Path,
    x: np.ndarray,
    y: np.ndarray,
    x_dense: np.ndarray,
    y_dense: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(x, y, s=16)
    plt.plot(x_dense, y_dense)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True, help="input csv path")
    ap.add_argument("--sigmas", type=str, required=True, help='comma-separated sigma tags, e.g. "s020,s025,s030"')
    ap.add_argument("--qps", type=str, required=True, help='comma-separated qps, e.g. "22,27,32,37"')
    ap.add_argument("--x_col", type=str, default="abs_reduction", help="x-axis column name")
    ap.add_argument("--clip_col", type=str, default="clip_name", help="clip name column")
    ap.add_argument("--out_dir", type=str, default="", help="default: same folder as csv")
    ap.add_argument("--make_plots", action="store_true", help="save scatter+fit plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = parse_str_list(args.sigmas)
    qps = parse_int_list(args.qps)

    df = pd.read_csv(csv_path)

    if args.clip_col not in df.columns:
        raise KeyError(f"Missing clip column: {args.clip_col}")
    if args.x_col not in df.columns:
        raise KeyError(f"Missing x column: {args.x_col}")

    long_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []

    x_all = pd.to_numeric(df[args.x_col], errors="coerce").to_numpy(np.float64)

    for sigma in sigmas:
        for qp in qps:
            delta_col = col_delta_kbps(sigma, qp)
            psnr_col = col_psnr(sigma, qp)

            if delta_col not in df.columns:
                print(f"[WARN] missing column: {delta_col}")
                continue
            if psnr_col not in df.columns:
                print(f"[WARN] missing column: {psnr_col}")
                continue

            # --------------------------------------------------
            # delta_kbps : logistic
            # --------------------------------------------------
            y_delta = pd.to_numeric(df[delta_col], errors="coerce").to_numpy(np.float64)
            mask_delta = np.isfinite(x_all) & np.isfinite(y_delta)

            if np.sum(mask_delta) >= 6:
                x = x_all[mask_delta]
                y = y_delta[mask_delta]

                try:
                    popt, yhat, r2 = fit_logistic4(x, y)
                    a, b, c, d = [float(v) for v in popt]

                    long_rows.append({
                        "sigma": sigma,
                        "qp": qp,
                        "metric": "delta_kbps",
                        "model": "logistic4",
                        "n_points": int(len(x)),
                        "r2": r2,
                        "param1_name": "a",
                        "param1_value": a,
                        "param2_name": "b",
                        "param2_value": b,
                        "param3_name": "c",
                        "param3_value": c,
                        "param4_name": "d",
                        "param4_value": d,
                    })

                    wide_rows.append({
                        "sigma": sigma,
                        "qp": qp,
                        "metric": "delta_kbps",
                        "model": "logistic4",
                        "n_points": int(len(x)),
                        "r2": r2,
                        "a": a,
                        "b": b,
                        "c": c,
                        "d": d,
                        "p2": np.nan,
                        "p1": np.nan,
                        "p0": np.nan,
                    })

                    if args.make_plots:
                        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                        y_dense = logistic4(x_dense, *popt)
                        plot_scatter_and_fit(
                            out_png=out_dir / "plots" / f"{sigma}_qp{qp}_delta_kbps.png",
                            x=x,
                            y=y,
                            x_dense=x_dense,
                            y_dense=y_dense,
                            title=f"{sigma} qp{qp} | delta_kbps vs {args.x_col}",
                            xlabel=args.x_col,
                            ylabel=delta_col,
                        )
                except Exception as e:
                    print(f"[WARN] logistic fit failed for {sigma} qp{qp}: {e}")
            else:
                print(f"[WARN] not enough valid points for {delta_col}")

            # --------------------------------------------------
            # psnrY_blur_deblur : quadratic
            # --------------------------------------------------
            y_psnr = pd.to_numeric(df[psnr_col], errors="coerce").to_numpy(np.float64)
            mask_psnr = np.isfinite(x_all) & np.isfinite(y_psnr)

            if np.sum(mask_psnr) >= 3:
                x = x_all[mask_psnr]
                y = y_psnr[mask_psnr]

                try:
                    coeffs, yhat, r2 = fit_poly2(x, y)
                    p2, p1, p0 = [float(v) for v in coeffs]

                    long_rows.append({
                        "sigma": sigma,
                        "qp": qp,
                        "metric": "psnrY_blur_deblur",
                        "model": "poly2",
                        "n_points": int(len(x)),
                        "r2": r2,
                        "param1_name": "p2",
                        "param1_value": p2,
                        "param2_name": "p1",
                        "param2_value": p1,
                        "param3_name": "p0",
                        "param3_value": p0,
                        "param4_name": "",
                        "param4_value": np.nan,
                    })

                    wide_rows.append({
                        "sigma": sigma,
                        "qp": qp,
                        "metric": "psnrY_blur_deblur",
                        "model": "poly2",
                        "n_points": int(len(x)),
                        "r2": r2,
                        "a": np.nan,
                        "b": np.nan,
                        "c": np.nan,
                        "d": np.nan,
                        "p2": p2,
                        "p1": p1,
                        "p0": p0,
                    })

                    if args.make_plots:
                        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)
                        y_dense = np.polyval(coeffs, x_dense)
                        plot_scatter_and_fit(
                            out_png=out_dir / "plots" / f"{sigma}_qp{qp}_psnrY_blur_deblur.png",
                            x=x,
                            y=y,
                            x_dense=x_dense,
                            y_dense=y_dense,
                            title=f"{sigma} qp{qp} | psnrY_blur_deblur vs {args.x_col}",
                            xlabel=args.x_col,
                            ylabel=psnr_col,
                        )
                except Exception as e:
                    print(f"[WARN] poly2 fit failed for {sigma} qp{qp}: {e}")
            else:
                print(f"[WARN] not enough valid points for {psnr_col}")

    if not long_rows:
        raise RuntimeError("No successful fitting results were produced.")

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)

    # Sort for stable downstream use
    df_long = df_long.sort_values(["sigma", "qp", "metric"]).reset_index(drop=True)
    df_wide = df_wide.sort_values(["sigma", "qp", "metric"]).reset_index(drop=True)

    out_long = out_dir / "fit_params_long.csv"
    out_wide = out_dir / "fit_params_wide.csv"

    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")


if __name__ == "__main__":
    main()
