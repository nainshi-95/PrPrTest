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


# ============================================================
# BD-rate (cubic polynomial, standard style)
# ============================================================
def bd_rate_cubic(anchor_rate, anchor_psnr, test_rate, test_psnr) -> float:
    """
    Standard BD-rate with cubic polyfit:
      log(rate) = f(psnr), integrate over common PSNR interval.

    Returns:
      percent. Negative is better.
      NaN if invalid / overflow.
    """
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
# YUV420p10le loading + SI/TI
# ============================================================
def read_yuv420p10le_clip(path: Path, w=128, h=128, frames=33):
    """
    Returns:
      Y: (T,H,W)
      U: (T,H/2,W/2)
      V: (T,H/2,W/2)
    """
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
    """
    SI: mean of per-frame variance
    TI: mean of mean(diff^2) over time
    """
    T = channel.shape[0]
    si = float(np.mean([np.var(channel[t]) for t in range(T)]))
    if T < 2:
        return si, np.nan
    ti = float(np.mean([np.mean((channel[t] - channel[t - 1]) ** 2) for t in range(1, T)]))
    return si, ti


# ============================================================
# Proxy sigma loading
# ============================================================
def load_proxy_best_sigma_map(
    proxy_sigma_csv: Path,
    qps: List[int],
    sigma_col_prefix: str,
) -> Dict[str, Dict[int, float]]:
    """
    Expected columns:
      clip_name or clip_id
      <sigma_col_prefix>22
      <sigma_col_prefix>27
      ...

    Example prefix:
      pred_best_sigma_qp
    then columns are:
      pred_best_sigma_qp22, pred_best_sigma_qp27, ...

    Returns:
      out[clip_id][qp] = sigma(float)
    """
    df = pd.read_csv(proxy_sigma_csv)
    df.columns = [c.strip() for c in df.columns]

    clip_key = None
    for cand in ["clip_id", "clip_name"]:
        if cand in df.columns:
            clip_key = cand
            break
    if clip_key is None:
        raise KeyError("proxy sigma csv must contain clip_id or clip_name")

    out: Dict[str, Dict[int, float]] = {}

    for _, row in df.iterrows():
        clip_id = str(row[clip_key]).strip()
        if not clip_id:
            continue

        out.setdefault(clip_id, {})
        for qp in qps:
            col = f"{sigma_col_prefix}{qp}"
            if col not in df.columns:
                continue
            v = safe_float(row[col])
            if np.isfinite(v):
                out[clip_id][qp] = float(v)

    return out


# ============================================================
# Build mixed curve from external sigma selection
# ============================================================
def build_curve_from_external_sigmas(
    per_sigma: Dict[float, Dict[str, np.ndarray]],
    chosen_sigma_by_qp: Dict[int, float],
    qp_list: List[int],
) -> Tuple[List[float], Dict[str, np.ndarray]]:
    """
    Build mixed curve using externally provided sigma for each QP.
    Returns:
      used_sigmas_in_qp_order
      out curve dict
    """
    n = len(qp_list)
    out = {
        "kbps_blur": np.full(n, np.nan, dtype=np.float64),
    }
    for ch in ["Y", "U", "V"]:
        out[f"psnr_{ch}_deblur"] = np.full(n, np.nan, dtype=np.float64)

    used_sigmas = []

    available_sigmas = sorted(per_sigma.keys())

    for qi, qp in enumerate(qp_list):
        s = chosen_sigma_by_qp.get(qp, np.nan)
        used_sigmas.append(s)

        if not np.isfinite(s):
            continue

        # exact match preferred
        selected_sigma = None
        for ss in available_sigmas:
            if abs(float(ss) - float(s)) < 1e-12:
                selected_sigma = ss
                break

        # fallback: nearest sigma if exact missing
        if selected_sigma is None and len(available_sigmas) > 0:
            selected_sigma = min(available_sigmas, key=lambda z: abs(float(z) - float(s)))

        if selected_sigma is None:
            continue

        d = per_sigma[selected_sigma]
        out["kbps_blur"][qi] = d["kbps_blur"][qi]
        for ch in ["Y", "U", "V"]:
            out[f"psnr_{ch}_deblur"][qi] = d[f"psnr_{ch}_deblur"][qi]

        used_sigmas[-1] = float(selected_sigma)

    return used_sigmas, out


# ============================================================
# Match enhance points by nearest bitrate
# ============================================================
def build_matched_enh_curve_for_base_qps(
    ref: Dict[str, np.ndarray],
    test_curve: Dict[str, np.ndarray],
    qp_list: List[int],
    base_qp_list: List[int],
) -> Dict[str, np.ndarray]:
    """
    For each base_qp test point, choose the enhance point
    (from all available QPs) whose bitrate is closest to that test bitrate.

    Anchor for final BD-rate becomes this matched enhance curve.
    """
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
        plt.plot(rate_deb[m2], psnr_deb[m2], marker="o", label="proxy_selected_deblur")

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
                    help='Comma-separated sigma list, e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"')
    ap.add_argument("--gt_yuv_root", type=str, required=True,
                    help="Folder containing all GT yuv clips named <clip_id>.yuv")
    ap.add_argument("--proxy_sigma_csv", type=str, required=True,
                    help="CSV containing proxy-selected sigma per clip and qp")
    ap.add_argument("--proxy_sigma_col_prefix", type=str, default="pred_best_sigma_qp",
                    help='Prefix for proxy sigma columns, e.g. "pred_best_sigma_qp" -> pred_best_sigma_qp22')
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--qps", type=str, required=True,
                    help='All available QPs in ascending order, e.g. "22,23,24,25,27,29,32,34,37"')
    ap.add_argument("--base_qps", type=str, default="22,27,32,37",
                    help='QPs used for final BD-rate, e.g. "22,27,32,37"')

    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--frames", type=int, default=33)

    args = ap.parse_args()

    sigma_list = parse_float_list(args.sigmas)
    qp_list = parse_int_list(args.qps)
    base_qp_list = parse_int_list(args.base_qps)

    root = Path(args.root)
    gt_yuv_root = Path(args.gt_yuv_root)
    proxy_sigma_csv = Path(args.proxy_sigma_csv)
    out_root = Path(args.out)

    ensure_dir(out_root)
    plot_root = out_root / "per_clip_plots"
    scatter_root = out_root / "scatter"
    ensure_dir(plot_root)
    ensure_dir(scatter_root)

    # ------------------------------------------------------------
    # Load proxy sigma selections
    # ------------------------------------------------------------
    proxy_sigma_map = load_proxy_best_sigma_map(
        proxy_sigma_csv=proxy_sigma_csv,
        qps=base_qp_list,
        sigma_col_prefix=args.proxy_sigma_col_prefix,
    )

    # ------------------------------------------------------------
    # Load all sigma folders
    # clip_map[(class, seq, clip)][sigma] = arrays
    # ------------------------------------------------------------
    clip_map: Dict[Tuple[str, str, str], Dict[float, Dict[str, np.ndarray]]] = {}

    for s in sorted(sigma_list):
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

        if clip_id not in proxy_sigma_map:
            print(f"[WARN] No proxy sigma entry for clip: {clip_id}")
            continue

        chosen_sigma_by_qp = proxy_sigma_map[clip_id]

        ref = per_sigma[sigmas_here[0]]

        used_sigmas, proxy_curve = build_curve_from_external_sigmas(
            per_sigma=per_sigma,
            chosen_sigma_by_qp=chosen_sigma_by_qp,
            qp_list=qp_list,
        )

        matched_enh = build_matched_enh_curve_for_base_qps(
            ref=ref,
            test_curve=proxy_curve,
            qp_list=qp_list,
            base_qp_list=base_qp_list,
        )

        # indices of base_qps in full qp_list
        base_idx = np.array([qp_list.index(q) for q in base_qp_list if q in qp_list], dtype=int)

        # BD-rate for Y/U/V : proxy-selected deblur vs matched enhance
        bdr = {}
        for ch in ["Y", "U", "V"]:
            bdr[ch] = bd_rate_cubic(
                anchor_rate=matched_enh["kbps_enh"],
                anchor_psnr=matched_enh[f"psnr_{ch}_enh"],
                test_rate=proxy_curve["kbps_blur"][base_idx],
                test_psnr=proxy_curve[f"psnr_{ch}_deblur"][base_idx],
            )

        # reference enhance BD-rate vs gt_codec
        bdr_enh = {}
        for ch in ["Y", "U", "V"]:
            bdr_enh[ch] = bd_rate_cubic(
                anchor_rate=ref["kbps_gt"],
                anchor_psnr=ref[f"psnr_{ch}_gt_codec"],
                test_rate=ref["kbps_gt"],
                test_psnr=ref[f"psnr_{ch}_gt_enhance"],
            )

        # Plot Y/U/V
        clip_plot_dir = plot_root / seq_class / seq_name / clip_id
        ensure_dir(clip_plot_dir)

        combo_txt = ", ".join(
            [f"QP{qp}:{sigma_to_tag(float(s))}" for qp, s in zip(qp_list, used_sigmas) if np.isfinite(s)]
        )

        for ch in ["Y", "U", "V"]:
            plot_rd_curve(
                out_path=clip_plot_dir / f"rd_{ch}.png",
                title=f"{seq_class}/{seq_name}/{clip_id}\nProxy combo: {combo_txt}",
                rate_codec=ref["kbps_gt"],
                psnr_codec=ref[f"psnr_{ch}_gt_codec"],
                rate_enh=ref["kbps_gt"],
                psnr_enh=ref[f"psnr_{ch}_gt_enhance"],
                rate_deb=proxy_curve["kbps_blur"],
                psnr_deb=proxy_curve[f"psnr_{ch}_deblur"],
                y_label=f"PSNR-{ch} (dB)",
            )

        # SI/TI
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
        }

        for qp, s in zip(qp_list, used_sigmas):
            row[f"proxy_sigma_qp{qp}"] = "" if not np.isfinite(s) else sigma_to_tag(float(s))

        row["base_qp_list"] = ",".join(map(str, base_qp_list))
        row["matched_enh_qps"] = ",".join(map(str, matched_enh["qp_enh_match"]))

        row["bdr_proxy_Y"] = bdr["Y"]
        row["bdr_proxy_U"] = bdr["U"]
        row["bdr_proxy_V"] = bdr["V"]

        row["bdr_enh_Y"] = bdr_enh["Y"]
        row["bdr_enh_U"] = bdr_enh["U"]
        row["bdr_enh_V"] = bdr_enh["V"]

        row["SI_Y"] = si_y
        row["TI_Y"] = ti_y
        row["SI_U"] = si_u
        row["TI_U"] = ti_u
        row["SI_V"] = si_v
        row["TI_V"] = ti_v

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid clips were processed.")

    summary_csv = out_root / "per_clip_proxy_summary.csv"
    df.to_csv(summary_csv, index=False)

    # ------------------------------------------------------------
    # Scatter plots: SI/TI vs proxy BD-rate, per channel
    # ------------------------------------------------------------
    for ch in ["Y", "U", "V"]:
        x_si = df[f"SI_{ch}"].to_numpy(dtype=np.float64)
        x_ti = df[f"TI_{ch}"].to_numpy(dtype=np.float64)
        y_bd = df[f"bdr_proxy_{ch}"].to_numpy(dtype=np.float64)

        m_si = np.isfinite(x_si) & np.isfinite(y_bd)
        m_ti = np.isfinite(x_ti) & np.isfinite(y_bd)

        if np.any(m_si):
            scatter_plot(
                out_path=scatter_root / f"SI_vs_BDR_proxy_{ch}.png",
                x=x_si[m_si],
                y=y_bd[m_si],
                xlabel=f"SI_{ch}",
                ylabel=f"Proxy BD-rate {ch} (%)",
                title=f"SI vs Proxy BD-rate ({ch})",
            )

        if np.any(m_ti):
            scatter_plot(
                out_path=scatter_root / f"TI_vs_BDR_proxy_{ch}.png",
                x=x_ti[m_ti],
                y=y_bd[m_ti],
                xlabel=f"TI_{ch}",
                ylabel=f"Proxy BD-rate {ch} (%)",
                title=f"TI vs Proxy BD-rate ({ch})",
            )

    print(f"[OK] summary csv: {summary_csv}")
    print(f"[OK] per-clip plots: {plot_root}")
    print(f"[OK] scatter plots: {scatter_root}")


if __name__ == "__main__":
    main()
