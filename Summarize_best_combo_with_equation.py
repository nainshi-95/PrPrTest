#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polyutils import RankWarning


# ============================================================
# Basic utils
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sigma_to_tag(s: float) -> str:
    return f"s{int(round(s * 100)):03d}"


def tag_to_sigma(tag: str) -> float:
    tag = tag.strip().lower()
    if not tag.startswith("s"):
        raise ValueError(f"Bad sigma tag: {tag}")
    return float(int(tag[1:])) / 100.0


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def pick_col(df: pd.DataFrame, name: str) -> str:
    cols = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key not in cols:
        raise KeyError(f"Missing column '{name}'. Have={list(df.columns)}")
    return cols[key]


def parse_int_list(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("Empty integer list.")
    return sorted(out)


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("Empty float list.")
    return sorted(out)


def parse_lambda_map(s: str) -> Dict[int, float]:
    """
    Example:
      22=0.01,27=0.03,32=0.1,37=0.3
    """
    out: Dict[int, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad lambda item: {part}")
        k, v = part.split("=", 1)
        out[int(k.strip())] = float(v.strip())
    if not out:
        raise ValueError("Empty lambda map.")
    return out


# ============================================================
# BD-rate
# ============================================================
def bd_rate_cubic(anchor_rate, anchor_psnr, test_rate, test_psnr) -> float:
    a_r = np.asarray(anchor_rate, dtype=np.float64)
    a_p = np.asarray(anchor_psnr, dtype=np.float64)
    t_r = np.asarray(test_rate, dtype=np.float64)
    t_p = np.asarray(test_psnr, dtype=np.float64)

    a_mask = np.isfinite(a_r) & np.isfinite(a_p) & (a_r > 0)
    t_mask = np.isfinite(t_r) & np.isfinite(t_p) & (t_r > 0)
    a_r, a_p = a_r[a_mask], a_p[a_mask]
    t_r, t_p = t_r[t_mask], t_p[t_mask]

    if len(a_r) < 2 or len(t_r) < 2:
        return np.nan

    ai = np.argsort(a_p)
    ti = np.argsort(t_p)
    a_p, a_r = a_p[ai], a_r[ai]
    t_p, t_r = t_p[ti], t_r[ti]

    p_min = max(np.min(a_p), np.min(t_p))
    p_max = min(np.max(a_p), np.max(t_p))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return np.nan

    la = np.log(a_r)
    lt = np.log(t_r)

    deg = 3
    if len(a_p) < 4 or len(t_p) < 4:
        deg = min(3, len(a_p) - 1, len(t_p) - 1)
        if deg < 1:
            return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        try:
            pa = np.polyfit(a_p, la, deg)
            pt = np.polyfit(t_p, lt, deg)
        except Exception:
            return np.nan

    ia = np.polyint(pa)
    it = np.polyint(pt)

    inta = np.polyval(ia, p_max) - np.polyval(ia, p_min)
    intt = np.polyval(it, p_max) - np.polyval(it, p_min)

    avg_diff = (intt - inta) / (p_max - p_min)
    if not np.isfinite(avg_diff):
        return np.nan

    exp_val = np.exp(avg_diff)
    if not np.isfinite(exp_val):
        return np.nan

    return float((exp_val - 1.0) * 100.0)


# ============================================================
# Model equations
# ============================================================
def logistic4(x: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray | float:
    z = np.clip(-b * (np.asarray(x) - c), -60.0, 60.0)
    return d + (a - d) / (1.0 + np.exp(z))


def poly2(x: np.ndarray | float, p2: float, p1: float, p0: float) -> np.ndarray | float:
    x = np.asarray(x)
    return p2 * x * x + p1 * x + p0


# ============================================================
# CSV loading
# ============================================================
def collect_rd_csvs(sigma_root: Path) -> List[Path]:
    per_clip = sigma_root / "per_clip"
    if not per_clip.exists():
        return []
    return list(per_clip.rglob("rd_points.csv"))


def load_rd_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clip_meta_from_path(csv_path: Path) -> Tuple[str, str, str]:
    # .../per_clip/<class>/<sequence>/<clip>/rd_points.csv
    parts = csv_path.parts
    seq_class = parts[-4]
    seq_name = parts[-3]
    clip_id = parts[-2]
    return seq_class, seq_name, clip_id


def extract_arrays(df: pd.DataFrame, qp_list: List[int]) -> Dict[str, np.ndarray]:
    c_qp = pick_col(df, "qp")
    qp_to_row = {}
    for i in range(len(df)):
        qp = int(round(safe_float(df.loc[i, c_qp])))
        qp_to_row[qp] = i

    def arr(colname: str) -> np.ndarray:
        c = pick_col(df, colname)
        out = []
        for qp in qp_list:
            row = qp_to_row.get(qp, None)
            out.append(np.nan if row is None else safe_float(df.loc[row, c]))
        return np.asarray(out, dtype=np.float64)

    out = {
        "kbps_gt": arr("kbps_gt"),
        "kbps_blur": arr("kbps_blur"),
        "bpp_gt": arr("bpp_gt"),
        "bpp_blur": arr("bpp_blur"),
    }

    def get_psnr_col(ch: str, kind_primary: str, kind_alt: Optional[str] = None) -> np.ndarray:
        cand = [f"psnr_{ch}_{kind_primary}"]
        if kind_alt is not None:
            cand.append(f"psnr_{ch}_{kind_alt}")
        for name in cand:
            try:
                return arr(name)
            except Exception:
                pass
        raise KeyError(f"Missing PSNR column candidates: {cand}")

    for ch in ["Y", "U", "V"]:
        out[f"psnr_{ch}_gt_codec"]   = get_psnr_col(ch, "gt_codec")
        out[f"psnr_{ch}_gt_enhance"] = get_psnr_col(ch, "gt_enhance", "gt_enh")
        out[f"psnr_{ch}_blur"]       = get_psnr_col(ch, "blur", "blur_codec")
        out[f"psnr_{ch}_deblur"]     = get_psnr_col(ch, "deblur", "blur_deblur")

    return out


# ============================================================
# YUV loading + SI/TI
# ============================================================
def read_yuv420p10le_clip(path: Path, w=128, h=128, frames=33):
    w2, h2 = w // 2, h // 2
    y_n = w * h
    uv_n = w2 * h2
    per_frame = y_n + uv_n + uv_n
    total = per_frame * frames

    data = np.fromfile(path, dtype=np.uint16, count=total)
    if data.size != total:
        raise RuntimeError(f"Unexpected sample count in {path}: got {data.size}, expected {total}")

    data = data.astype(np.float32)

    Y = np.empty((frames, h, w), dtype=np.float32)
    U = np.empty((frames, h2, w2), dtype=np.float32)
    V = np.empty((frames, h2, w2), dtype=np.float32)

    idx = 0
    for t in range(frames):
        Y[t] = data[idx:idx + y_n].reshape(h, w)
        idx += y_n
        U[t] = data[idx:idx + uv_n].reshape(h2, w2)
        idx += uv_n
        V[t] = data[idx:idx + uv_n].reshape(h2, w2)
        idx += uv_n

    return Y, U, V


def compute_si_ti(channel: np.ndarray) -> Tuple[float, float]:
    T = channel.shape[0]
    si = float(np.mean([np.var(channel[t]) for t in range(T)]))
    if T < 2:
        return si, np.nan
    ti = float(np.mean([np.mean((channel[t] - channel[t - 1]) ** 2) for t in range(1, T)]))
    return si, ti


# ============================================================
# Fit params loading
# ============================================================
def load_fit_param_map(fit_csv: Path) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    """
    Input: fit_params_wide.csv
    Key:
      (sigma_tag, qp, metric)
    metric:
      - delta_kbps
      - psnrY_blur_deblur   # name is kept, but values are assumed to be MSE-model params
    """
    df = pd.read_csv(fit_csv)
    df.columns = [c.strip() for c in df.columns]

    required = ["sigma", "qp", "metric", "model", "r2", "a", "b", "c", "d", "p2", "p1", "p0"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column in fit csv: {c}")

    out: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        sigma = str(row["sigma"]).strip()
        qp = int(row["qp"])
        metric = str(row["metric"]).strip()
        out[(sigma, qp, metric)] = {
            "model": str(row["model"]).strip(),
            "r2": safe_float(row["r2"]),
            "a": safe_float(row["a"]),
            "b": safe_float(row["b"]),
            "c": safe_float(row["c"]),
            "d": safe_float(row["d"]),
            "p2": safe_float(row["p2"]),
            "p1": safe_float(row["p1"]),
            "p0": safe_float(row["p0"]),
        }
    return out


def eval_fit(
    fit_map: Dict[Tuple[str, int, str], Dict[str, float]],
    sigma_tag: str,
    qp: int,
    metric: str,
    x: float,
) -> float:
    key = (sigma_tag, qp, metric)
    if key not in fit_map:
        return np.nan

    info = fit_map[key]
    model = info["model"]

    if model == "logistic4":
        return float(logistic4(x, info["a"], info["b"], info["c"], info["d"]))
    elif model == "poly2":
        return float(poly2(x, info["p2"], info["p1"], info["p0"]))
    else:
        return np.nan


# ============================================================
# abs_reduction loading
# ============================================================
def load_abs_reduction_map(
    csv_path: Path,
    clip_col_candidates: List[str],
    x_col: str,
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    clip_col = None
    for cand in clip_col_candidates:
        if cand in df.columns:
            clip_col = cand
            break
    if clip_col is None:
        raise KeyError(f"Could not find clip column from {clip_col_candidates}")
    if x_col not in df.columns:
        raise KeyError(f"Missing x column: {x_col}")

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        clip_id = str(row[clip_col]).strip()
        x = safe_float(row[x_col])
        if clip_id and np.isfinite(x):
            out[clip_id] = float(x)
    return out


# ============================================================
# Selection
# ============================================================
def select_best_sigmas_by_lambda(
    sigma_tags: List[str],
    qps: List[int],
    lambda_map: Dict[int, float],
    abs_reduction: float,
    fit_map: Dict[Tuple[str, int, str], Dict[str, float]],
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]]]:
    """
    D = metric 'psnrY_blur_deblur' from fit csv, but interpreted as MSE
    R = metric 'delta_kbps'
    J = D + lambda * R
    """
    best_sigma_by_qp: Dict[int, str] = {}
    detail_by_qp: Dict[int, Dict[str, float]] = {}

    for qp in qps:
        lam = lambda_map.get(qp, None)
        if lam is None:
            continue

        best_j = np.inf
        best_sigma = None
        best_d = np.nan
        best_r = np.nan

        for sigma_tag in sigma_tags:
            r = eval_fit(fit_map, sigma_tag, qp, "delta_kbps", abs_reduction)
            d = eval_fit(fit_map, sigma_tag, qp, "psnrY_blur_deblur", abs_reduction)  # actually MSE params

            if not (np.isfinite(r) and np.isfinite(d)):
                continue

            j = d + lam * r
            if np.isfinite(j) and j < best_j:
                best_j = j
                best_sigma = sigma_tag
                best_d = d
                best_r = r

        if best_sigma is not None:
            best_sigma_by_qp[qp] = best_sigma
            detail_by_qp[qp] = {
                "pred_D": best_d,
                "pred_R": best_r,
                "pred_J": best_j,
                "lambda": lam,
            }

    return best_sigma_by_qp, detail_by_qp


# ============================================================
# Build selected real curve from actual rd_points.csv
# ============================================================
def build_curve_from_selected_sigmas(
    per_sigma: Dict[float, Dict[str, np.ndarray]],
    chosen_sigma_tag_by_qp: Dict[int, str],
    qp_list: List[int],
) -> Tuple[List[float], Dict[str, np.ndarray]]:
    n = len(qp_list)
    out = {
        "kbps_blur": np.full(n, np.nan, dtype=np.float64),
    }
    for ch in ["Y", "U", "V"]:
        out[f"psnr_{ch}_deblur"] = np.full(n, np.nan, dtype=np.float64)

    used_sigmas: List[float] = []
    available = sorted(per_sigma.keys())

    for qi, qp in enumerate(qp_list):
        tag = chosen_sigma_tag_by_qp.get(qp, None)
        if tag is None:
            used_sigmas.append(np.nan)
            continue

        s = tag_to_sigma(tag)
        used_sigmas.append(s)

        selected_sigma = None
        for ss in available:
            if abs(float(ss) - s) < 1e-12:
                selected_sigma = ss
                break

        if selected_sigma is None and len(available) > 0:
            selected_sigma = min(available, key=lambda z: abs(float(z) - s))

        if selected_sigma is None:
            continue

        d = per_sigma[selected_sigma]
        out["kbps_blur"][qi] = d["kbps_blur"][qi]
        for ch in ["Y", "U", "V"]:
            out[f"psnr_{ch}_deblur"][qi] = d[f"psnr_{ch}_deblur"][qi]

        used_sigmas[-1] = float(selected_sigma)

    return used_sigmas, out


# ============================================================
# Match enhance by nearest bitrate
# ============================================================
def build_matched_enh_curve_for_base_qps(
    ref: Dict[str, np.ndarray],
    test_curve: Dict[str, np.ndarray],
    qp_list: List[int],
    base_qp_list: List[int],
) -> Dict[str, np.ndarray]:
    qp_to_idx = {qp: i for i, qp in enumerate(qp_list)}

    enh_rates_all = np.asarray(ref["kbps_gt"], dtype=np.float64)

    out = {
        "kbps_enh": [],
        "qp_test": [],
        "qp_enh_match": [],
    }
    for ch in ["Y", "U", "V"]:
        out[f"psnr_{ch}_enh"] = []
        out[f"psnr_{ch}_deblur"] = []

    valid_enh_idx = np.where(np.isfinite(enh_rates_all) & (enh_rates_all > 0))[0].tolist()

    for base_qp in base_qp_list:
        if base_qp not in qp_to_idx:
            continue
        qi = qp_to_idx[base_qp]

        deb_rate = float(test_curve["kbps_blur"][qi])
        if not (np.isfinite(deb_rate) and deb_rate > 0):
            continue

        best_j = None
        best_abs = np.inf
        for j in valid_enh_idx:
            rr = float(enh_rates_all[j])
            dd = abs(rr - deb_rate)
            if dd < best_abs:
                best_abs = dd
                best_j = j

        if best_j is None:
            continue

        out["kbps_enh"].append(float(ref["kbps_gt"][best_j]))
        out["qp_test"].append(base_qp)
        out["qp_enh_match"].append(qp_list[best_j])

        for ch in ["Y", "U", "V"]:
            out[f"psnr_{ch}_enh"].append(float(ref[f"psnr_{ch}_gt_enhance"][best_j]))
            out[f"psnr_{ch}_deblur"].append(float(test_curve[f"psnr_{ch}_deblur"][qi]))

    for k in list(out.keys()):
        if k.startswith("psnr_") or k == "kbps_enh":
            out[k] = np.asarray(out[k], dtype=np.float64)

    return out


# ============================================================
# Plotting
# ============================================================
def plot_rd_curve(
    out_path: Path,
    title: str,
    rate_codec: np.ndarray,
    psnr_codec: np.ndarray,
    rate_enh: np.ndarray,
    psnr_enh: np.ndarray,
    rate_deb: np.ndarray,
    psnr_deb: np.ndarray,
    y_label: str,
):
    m0 = np.isfinite(rate_codec) & np.isfinite(psnr_codec) & (rate_codec > 0)
    m1 = np.isfinite(rate_enh)   & np.isfinite(psnr_enh)   & (rate_enh > 0)
    m2 = np.isfinite(rate_deb)   & np.isfinite(psnr_deb)   & (rate_deb > 0)

    fig = plt.figure()
    if np.any(m0):
        plt.plot(rate_codec[m0], psnr_codec[m0], marker="o", label="gt_codec")
    if np.any(m1):
        plt.plot(rate_enh[m1], psnr_enh[m1], marker="o", label="gt_enhance")
    if np.any(m2):
        plt.plot(rate_deb[m2], psnr_deb[m2], marker="o", label="selected_deblur")

    plt.xlabel("kbps")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
):
    fig = plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True,
                    help="Parent directory containing prefix_sXXX_postfix folders")
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--postfix", type=str, required=True)
    ap.add_argument("--sigmas", type=str, required=True,
                    help='Comma-separated sigma list, e.g. "0.20,0.25,0.30"')

    ap.add_argument("--fit_csv", type=str, required=True,
                    help="fit_params_wide.csv path")
    ap.add_argument("--abs_csv", type=str, required=True,
                    help="CSV containing clip_name/clip_id and abs_reduction")
    ap.add_argument("--abs_clip_cols", type=str, default="clip_name,clip_id",
                    help="candidate clip columns in abs csv")
    ap.add_argument("--abs_col", type=str, default="abs_reduction",
                    help="abs reduction column name")

    ap.add_argument("--gt_yuv_root", type=str, required=True,
                    help="Folder containing GT yuv clips named <clip_id>.yuv")
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--qps", type=str, required=True,
                    help='All available QPs, e.g. "22,23,24,25,27,29,32,34,37"')
    ap.add_argument("--base_qps", type=str, default="22,27,32,37",
                    help='QPs used for final BD-rate')
    ap.add_argument("--lambdas", type=str, required=True,
                    help='QP-lambda map, e.g. "22=0.01,27=0.03,32=0.1,37=0.3"')

    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--frames", type=int, default=33)

    args = ap.parse_args()

    sigma_vals = parse_float_list(args.sigmas)
    sigma_tags = [sigma_to_tag(s) for s in sigma_vals]
    qp_list = parse_int_list(args.qps)
    base_qp_list = parse_int_list(args.base_qps)
    lambda_map = parse_lambda_map(args.lambdas)

    root = Path(args.root)
    gt_yuv_root = Path(args.gt_yuv_root)
    out_root = Path(args.out)
    fit_csv = Path(args.fit_csv)
    abs_csv = Path(args.abs_csv)

    ensure_dir(out_root)
    plot_root = out_root / "per_clip_plots"
    scatter_root = out_root / "scatter"
    ensure_dir(plot_root)
    ensure_dir(scatter_root)

    fit_map = load_fit_param_map(fit_csv)
    abs_map = load_abs_reduction_map(
        abs_csv,
        clip_col_candidates=[x.strip() for x in args.abs_clip_cols.split(",") if x.strip()],
        x_col=args.abs_col,
    )

    # ------------------------------------------------------------
    # Load all sigma folders
    # ------------------------------------------------------------
    clip_map: Dict[Tuple[str, str, str], Dict[float, Dict[str, np.ndarray]]] = {}

    for s in sorted(sigma_vals):
        tag = sigma_to_tag(s)
        sigma_dir = root / f"{args.prefix}_{tag}_{args.postfix}"
        if not sigma_dir.exists():
            print(f"[WARN] Missing sigma folder: {sigma_dir}")
            continue

        csvs = collect_rd_csvs(sigma_dir)
        for csv_path in csvs:
            try:
                df = load_rd_csv(csv_path)
                seq_class, seq_name, clip_id = clip_meta_from_path(csv_path)
                arrs = extract_arrays(df, qp_list)
                clip_map.setdefault((seq_class, seq_name, clip_id), {})[s] = arrs
            except Exception as e:
                print(f"[WARN] Failed to parse {csv_path}: {e}")

    if not clip_map:
        raise RuntimeError("No valid rd_points.csv found.")

    # ------------------------------------------------------------
    # Per-clip processing
    # ------------------------------------------------------------
    rows = []

    for (seq_class, seq_name, clip_id), per_sigma in sorted(clip_map.items()):
        sigmas_here = sorted(per_sigma.keys())
        if len(sigmas_here) == 0:
            continue

        if clip_id not in abs_map:
            print(f"[WARN] No abs_reduction entry for clip: {clip_id}")
            continue

        x_abs = abs_map[clip_id]

        # use first sigma folder as GT/enh reference
        ref = per_sigma[sigmas_here[0]]

        best_sigma_by_qp, sel_detail = select_best_sigmas_by_lambda(
            sigma_tags=[sigma_to_tag(s) for s in sigmas_here],
            qps=qp_list,
            lambda_map=lambda_map,
            abs_reduction=x_abs,
            fit_map=fit_map,
        )

        used_sigmas, selected_curve = build_curve_from_selected_sigmas(
            per_sigma=per_sigma,
            chosen_sigma_tag_by_qp=best_sigma_by_qp,
            qp_list=qp_list,
        )

        matched_enh = build_matched_enh_curve_for_base_qps(
            ref=ref,
            test_curve=selected_curve,
            qp_list=qp_list,
            base_qp_list=base_qp_list,
        )

        base_idx = np.array([qp_list.index(q) for q in base_qp_list if q in qp_list], dtype=int)

        bdr = {}
        for ch in ["Y", "U", "V"]:
            bdr[ch] = bd_rate_cubic(
                anchor_rate=matched_enh["kbps_enh"],
                anchor_psnr=matched_enh[f"psnr_{ch}_enh"],
                test_rate=selected_curve["kbps_blur"][base_idx],
                test_psnr=selected_curve[f"psnr_{ch}_deblur"][base_idx],
            )

        bdr_enh = {}
        for ch in ["Y", "U", "V"]:
            bdr_enh[ch] = bd_rate_cubic(
                anchor_rate=ref["kbps_gt"],
                anchor_psnr=ref[f"psnr_{ch}_gt_codec"],
                test_rate=ref["kbps_gt"],
                test_psnr=ref[f"psnr_{ch}_gt_enhance"],
            )

        clip_plot_dir = plot_root / seq_class / seq_name / clip_id
        ensure_dir(clip_plot_dir)

        combo_txt = ", ".join(
            [f"QP{qp}:{sigma_to_tag(float(s))}" for qp, s in zip(qp_list, used_sigmas) if np.isfinite(s)]
        )

        for ch in ["Y", "U", "V"]:
            plot_rd_curve(
                out_path=clip_plot_dir / f"rd_{ch}.png",
                title=f"{seq_class}/{seq_name}/{clip_id}\nSelected combo: {combo_txt}",
                rate_codec=ref["kbps_gt"],
                psnr_codec=ref[f"psnr_{ch}_gt_codec"],
                rate_enh=ref["kbps_gt"],
                psnr_enh=ref[f"psnr_{ch}_gt_enhance"],
                rate_deb=selected_curve["kbps_blur"],
                psnr_deb=selected_curve[f"psnr_{ch}_deblur"],
                y_label=f"PSNR-{ch} (dB)",
            )

        yuv_path = gt_yuv_root / f"{clip_id}.yuv"
        si_y = si_u = si_v = np.nan
        ti_y = ti_u = ti_v = np.nan
        if yuv_path.exists():
            try:
                Y, U, V = read_yuv420p10le_clip(
                    yuv_path,
                    w=args.width,
                    h=args.height,
                    frames=args.frames,
                )
                si_y, ti_y = compute_si_ti(Y)
                si_u, ti_u = compute_si_ti(U)
                si_v, ti_v = compute_si_ti(V)
            except Exception as e:
                print(f"[WARN] SI/TI failed for {yuv_path}: {e}")
        else:
            print(f"[WARN] Missing GT yuv: {yuv_path}")

        row = {
            "sequence_class": seq_class,
            "sequence_name": seq_name,
            "clip_id": clip_id,
            "abs_reduction": x_abs,
            "base_qp_list": ",".join(map(str, base_qp_list)),
            "matched_enh_qps": ",".join(map(str, matched_enh["qp_enh_match"])),
            "bdr_selected_Y": bdr["Y"],
            "bdr_selected_U": bdr["U"],
            "bdr_selected_V": bdr["V"],
            "bdr_enh_Y": bdr_enh["Y"],
            "bdr_enh_U": bdr_enh["U"],
            "bdr_enh_V": bdr_enh["V"],
            "SI_Y": si_y,
            "TI_Y": ti_y,
            "SI_U": si_u,
            "TI_U": ti_u,
            "SI_V": si_v,
            "TI_V": ti_v,
        }

        for qp in qp_list:
            tag = best_sigma_by_qp.get(qp, "")
            row[f"selected_sigma_qp{qp}"] = tag

            d = sel_detail.get(qp, {})
            row[f"pred_D_qp{qp}"] = d.get("pred_D", np.nan)
            row[f"pred_R_qp{qp}"] = d.get("pred_R", np.nan)
            row[f"pred_J_qp{qp}"] = d.get("pred_J", np.nan)
            row[f"lambda_qp{qp}"] = d.get("lambda", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid clips were processed.")

    summary_csv = out_root / "per_clip_lambda_selected_summary.csv"
    df.to_csv(summary_csv, index=False)

    for ch in ["Y", "U", "V"]:
        x_si = df[f"SI_{ch}"].to_numpy(dtype=np.float64)
        x_ti = df[f"TI_{ch}"].to_numpy(dtype=np.float64)
        y_bd = df[f"bdr_selected_{ch}"].to_numpy(dtype=np.float64)

        m_si = np.isfinite(x_si) & np.isfinite(y_bd)
        m_ti = np.isfinite(x_ti) & np.isfinite(y_bd)

        if np.any(m_si):
            scatter_plot(
                out_path=scatter_root / f"SI_vs_BDR_selected_{ch}.png",
                x=x_si[m_si],
                y=y_bd[m_si],
                xlabel=f"SI_{ch}",
                ylabel=f"Selected BD-rate {ch} (%)",
                title=f"SI vs Selected BD-rate ({ch})",
            )

        if np.any(m_ti):
            scatter_plot(
                out_path=scatter_root / f"TI_vs_BDR_selected_{ch}.png",
                x=x_ti[m_ti],
                y=y_bd[m_ti],
                xlabel=f"TI_{ch}",
                ylabel=f"Selected BD-rate {ch} (%)",
                title=f"TI vs Selected BD-rate ({ch})",
            )

    print(f"[OK] summary csv: {summary_csv}")
    print(f"[OK] per-clip plots: {plot_root}")
    print(f"[OK] scatter plots: {scatter_root}")


if __name__ == "__main__":
    main()
