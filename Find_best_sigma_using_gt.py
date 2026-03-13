import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================
def sigma_to_tag(sigma: float) -> str:
    v = int(round(sigma * 100))
    return f"s{v:03d}"


def parse_sigma_list(sigmas_str: str) -> List[float]:
    vals = []
    for x in sigmas_str.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def parse_qp_list(qps_str: str) -> List[int]:
    vals = []
    for x in qps_str.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def sigma_step(sigmas: List[float]) -> float:
    sigmas = sorted(sigmas)
    if len(sigmas) < 2:
        return 0.05
    diffs = [sigmas[i + 1] - sigmas[i] for i in range(len(sigmas) - 1)]
    return float(np.median(diffs))


def build_lambda_candidates(min_log10: float, max_log10: float, num: int) -> np.ndarray:
    return np.logspace(min_log10, max_log10, num=num)


# ============================================================
# GT best sigma loading
# ============================================================
def load_gt_best_sigma_map(best_sigma_csv: Path, qps: List[int]) -> Dict[Tuple[str, int], float]:
    """
    CSV columns:
      clip_id
      best_sigma_qp22
      best_sigma_qp27
      ...
    """
    df = pd.read_csv(best_sigma_csv)
    if "clip_id" not in df.columns:
        raise KeyError("best_sigma_csv must contain column 'clip_id'")

    out = {}
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"]).strip()
        if not clip_id:
            continue

        for qp in qps:
            col = f"best_sigma_qp{qp}"
            if col not in df.columns:
                continue
            v = safe_float(row[col])
            if v is not None:
                out[(clip_id, qp)] = v

    return out


# ============================================================
# Actual GT value lookup
# ============================================================
def get_actual_values_for_sigma(
    clip_row: pd.Series,
    sigma: float,
    qp: int,
    y_rate_key: str,
    y_dist_key: str,
):
    sigma_tag = sigma_to_tag(sigma)

    r_col = f"{y_rate_key}_{sigma_tag}_qp{qp}"
    d_col = f"{y_dist_key}_{sigma_tag}_qp{qp}"

    actual_r = safe_float(clip_row.get(r_col, ""))
    actual_d = safe_float(clip_row.get(d_col, ""))

    if actual_r is None or actual_d is None:
        return None

    return {
        "sigma": sigma,
        "sigma_tag": sigma_tag,
        "actual_delta_kbps": actual_r,
        "actual_delta_mse": actual_d,
    }


# ============================================================
# Actual-only lambda sweep with cost gap analysis
# ============================================================
def evaluate_lambda_for_qp_actual_only(
    unified_df: pd.DataFrame,
    clip_col: str,
    sigma_list: List[float],
    qp: int,
    y_rate_key: str,
    y_dist_key: str,
    gt_best_sigma_map: Dict[Tuple[str, int], float],
    lam: float,
) -> Tuple[Dict, List[Dict]]:
    """
    Uses ONLY actual delta_kbps / delta_mse from unified CSV.

    For each clip:
      predicted_best_sigma_from_actualJ =
          argmin_sigma [ actual_delta_mse + lambda * actual_delta_kbps ]

    Then compares that sigma to GT best sigma from best_sigma_csv.

    Also records:
      - actual best cost
      - second-best cost
      - best-second gap
      - cost at GT sigma from CSV
      - gap between GT-sigma cost and actual-best cost
    """
    step = sigma_step(sigma_list)

    errors = []
    soft_scores = []
    exact_hits = 0
    one_step_hits = 0
    two_step_hits = 0
    total = 0

    # gap / cost analysis
    best_second_gaps = []
    gap_gtcsv_to_actualbest_list = []
    large_error_gap_list = []

    per_clip_rows = []

    for _, row in unified_df.iterrows():
        clip_name = str(row[clip_col]).strip()
        if not clip_name:
            continue

        gt_sigma = gt_best_sigma_map.get((clip_name, qp), None)
        if gt_sigma is None:
            continue

        candidate_rows = []
        for sigma in sigma_list:
            actual = get_actual_values_for_sigma(
                clip_row=row,
                sigma=sigma,
                qp=qp,
                y_rate_key=y_rate_key,
                y_dist_key=y_dist_key,
            )
            if actual is None:
                continue

            J = actual["actual_delta_mse"] + lam * actual["actual_delta_kbps"]
            actual["actual_J"] = J
            candidate_rows.append(actual)

        if not candidate_rows:
            continue

        # sort by actual J
        candidate_rows_sorted = sorted(candidate_rows, key=lambda z: z["actual_J"])

        pred_best = candidate_rows_sorted[0]
        pred_sigma = float(pred_best["sigma"])
        pred_best_J = float(pred_best["actual_J"])

        # second-best
        if len(candidate_rows_sorted) >= 2:
            second_best = candidate_rows_sorted[1]
            second_best_sigma = float(second_best["sigma"])
            second_best_J = float(second_best["actual_J"])
            best_second_gap = second_best_J - pred_best_J
        else:
            second_best = None
            second_best_sigma = np.nan
            second_best_J = np.nan
            best_second_gap = np.nan

        # cost at GT sigma from CSV
        gt_sigma_row = None
        for c in candidate_rows:
            if abs(float(c["sigma"]) - float(gt_sigma)) < 1e-12:
                gt_sigma_row = c
                break

        if gt_sigma_row is not None:
            actual_J_at_gt_sigma_csv = float(gt_sigma_row["actual_J"])
            gap_gtcsv_to_actualbest = actual_J_at_gt_sigma_csv - pred_best_J
        else:
            actual_J_at_gt_sigma_csv = np.nan
            gap_gtcsv_to_actualbest = np.nan

        err = abs(pred_sigma - gt_sigma)
        errors.append(err)

        soft_score = math.exp(-err / max(step, 1e-12))
        soft_scores.append(soft_score)

        total += 1
        if err < 1e-12:
            exact_hits += 1
        if err <= step + 1e-12:
            one_step_hits += 1
        if err <= 2.0 * step + 1e-12:
            two_step_hits += 1

        if not np.isnan(best_second_gap):
            best_second_gaps.append(best_second_gap)

        if not np.isnan(gap_gtcsv_to_actualbest):
            gap_gtcsv_to_actualbest_list.append(gap_gtcsv_to_actualbest)

        is_large_sigma_error = bool(err > step + 1e-12)
        if is_large_sigma_error and not np.isnan(gap_gtcsv_to_actualbest):
            large_error_gap_list.append(gap_gtcsv_to_actualbest)

        per_clip_rows.append({
            "clip_name": clip_name,
            "qp": qp,
            "lambda": lam,

            "gt_best_sigma": gt_sigma,
            "pred_best_sigma_from_actualJ": pred_sigma,
            "abs_sigma_error": err,
            "soft_score": soft_score,
            "is_large_sigma_error_vs_gtcsv": int(is_large_sigma_error),

            "chosen_sigma_tag": pred_best["sigma_tag"],
            "chosen_actual_delta_kbps": pred_best["actual_delta_kbps"],
            "chosen_actual_delta_mse": pred_best["actual_delta_mse"],
            "chosen_actual_J": pred_best_J,

            "actual_best_sigma_from_actualJ": pred_sigma,
            "actual_best_J": pred_best_J,

            "actual_second_best_sigma": second_best_sigma,
            "actual_second_best_J": second_best_J,
            "actual_best_second_gap": best_second_gap,

            "actual_J_at_gt_sigma_csv": actual_J_at_gt_sigma_csv,
            "gap_gtcsv_to_actualbest": gap_gtcsv_to_actualbest,
        })

    if total == 0:
        summary = {
            "qp": qp,
            "lambda": lam,
            "num_clips": 0,

            "mae_sigma": np.nan,
            "rmse_sigma": np.nan,
            "mean_soft_score": np.nan,
            "exact_acc": np.nan,
            "within_1step_acc": np.nan,
            "within_2step_acc": np.nan,

            "mean_best_second_gap": np.nan,
            "median_best_second_gap": np.nan,
            "mean_gap_gtcsv_to_actualbest": np.nan,
            "median_gap_gtcsv_to_actualbest": np.nan,
            "mean_gap_for_large_sigma_error": np.nan,
            "median_gap_for_large_sigma_error": np.nan,
        }
        return summary, per_clip_rows

    errors = np.asarray(errors, dtype=np.float64)
    soft_scores = np.asarray(soft_scores, dtype=np.float64)

    summary = {
        "qp": qp,
        "lambda": lam,
        "num_clips": total,

        "mae_sigma": float(np.mean(errors)),
        "rmse_sigma": float(np.sqrt(np.mean(errors ** 2))),
        "mean_soft_score": float(np.mean(soft_scores)),
        "exact_acc": float(exact_hits / total),
        "within_1step_acc": float(one_step_hits / total),
        "within_2step_acc": float(two_step_hits / total),
    }

    if len(best_second_gaps) > 0:
        arr = np.asarray(best_second_gaps, dtype=np.float64)
        summary["mean_best_second_gap"] = float(np.mean(arr))
        summary["median_best_second_gap"] = float(np.median(arr))
    else:
        summary["mean_best_second_gap"] = np.nan
        summary["median_best_second_gap"] = np.nan

    if len(gap_gtcsv_to_actualbest_list) > 0:
        arr = np.asarray(gap_gtcsv_to_actualbest_list, dtype=np.float64)
        summary["mean_gap_gtcsv_to_actualbest"] = float(np.mean(arr))
        summary["median_gap_gtcsv_to_actualbest"] = float(np.median(arr))
    else:
        summary["mean_gap_gtcsv_to_actualbest"] = np.nan
        summary["median_gap_gtcsv_to_actualbest"] = np.nan

    if len(large_error_gap_list) > 0:
        arr = np.asarray(large_error_gap_list, dtype=np.float64)
        summary["mean_gap_for_large_sigma_error"] = float(np.mean(arr))
        summary["median_gap_for_large_sigma_error"] = float(np.median(arr))
    else:
        summary["mean_gap_for_large_sigma_error"] = np.nan
        summary["median_gap_for_large_sigma_error"] = np.nan

    return summary, per_clip_rows


def select_best_lambda(lambda_rows: List[Dict]) -> Optional[Dict]:
    """
    Primary: smallest MAE
    Secondary: highest soft score
    Tertiary: highest within_1step_acc
    """
    valid = [r for r in lambda_rows if r["num_clips"] > 0 and not np.isnan(r["mae_sigma"])]
    if not valid:
        return None

    valid.sort(
        key=lambda r: (
            r["mae_sigma"],
            -r["mean_soft_score"],
            -r["within_1step_acc"],
            -r["within_2step_acc"],
        )
    )
    return valid[0]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unified_csv", type=str, required=True)
    parser.add_argument("--best_sigma_csv", type=str, required=True)

    parser.add_argument(
        "--sigmas",
        type=str,
        required=True,
        help='e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"',
    )
    parser.add_argument("--qps", type=str, default="22,27,32,37")

    parser.add_argument("--clip_col", type=str, default="clip_name")
    parser.add_argument("--y_rate_key", type=str, default="delta_kbps")
    parser.add_argument("--y_dist_key", type=str, default="delta_mse")

    parser.add_argument("--lambda_min_log10", type=float, default=-10.0)
    parser.add_argument("--lambda_max_log10", type=float, default=0.0)
    parser.add_argument("--lambda_num", type=int, default=401)

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    unified_df = pd.read_csv(args.unified_csv)
    sigma_list = parse_sigma_list(args.sigmas)
    qps = parse_qp_list(args.qps)

    gt_best_sigma_map = load_gt_best_sigma_map(Path(args.best_sigma_csv), qps)

    lambda_candidates = build_lambda_candidates(
        min_log10=args.lambda_min_log10,
        max_log10=args.lambda_max_log10,
        num=args.lambda_num,
    )

    all_lambda_rows = []
    best_lambda_rows = []
    best_pred_rows = []

    for qp in qps:
        print(f"[INFO] Sweeping actual-only lambda for qp={qp} ...")

        qp_lambda_rows = []
        for lam in lambda_candidates:
            summary, _ = evaluate_lambda_for_qp_actual_only(
                unified_df=unified_df,
                clip_col=args.clip_col,
                sigma_list=sigma_list,
                qp=qp,
                y_rate_key=args.y_rate_key,
                y_dist_key=args.y_dist_key,
                gt_best_sigma_map=gt_best_sigma_map,
                lam=float(lam),
            )
            qp_lambda_rows.append(summary)
            all_lambda_rows.append(summary)

        best_lambda = select_best_lambda(qp_lambda_rows)
        if best_lambda is None:
            print(f"[WARN] No valid lambda found for qp={qp}")
            continue

        best_lambda_rows.append(best_lambda)

        print(
            f"[INFO] Best actual-only lambda for qp={qp}: {best_lambda['lambda']:.6e}, "
            f"MAE={best_lambda['mae_sigma']:.6f}, "
            f"1step={best_lambda['within_1step_acc']:.6f}, "
            f"best-second-gap(mean)={best_lambda['mean_best_second_gap']:.6e}, "
            f"gap_gt_to_best(mean)={best_lambda['mean_gap_gtcsv_to_actualbest']:.6e}"
        )

        pd.DataFrame(qp_lambda_rows).to_csv(
            output_dir / f"lambda_sweep_actual_only_qp{qp}.csv",
            index=False,
        )

        _, per_clip_rows = evaluate_lambda_for_qp_actual_only(
            unified_df=unified_df,
            clip_col=args.clip_col,
            sigma_list=sigma_list,
            qp=qp,
            y_rate_key=args.y_rate_key,
            y_dist_key=args.y_dist_key,
            gt_best_sigma_map=gt_best_sigma_map,
            lam=float(best_lambda["lambda"]),
        )
        best_pred_rows.extend(per_clip_rows)

    pd.DataFrame(all_lambda_rows).to_csv(output_dir / "lambda_sweep_actual_only_all.csv", index=False)
    pd.DataFrame(best_lambda_rows).to_csv(output_dir / "best_lambda_actual_only_per_qp.csv", index=False)
    pd.DataFrame(best_pred_rows).to_csv(output_dir / "best_sigma_actual_only_prediction_per_clip.csv", index=False)

    print(f"[INFO] Saved: {output_dir / 'lambda_sweep_actual_only_all.csv'}")
    print(f"[INFO] Saved: {output_dir / 'best_lambda_actual_only_per_qp.csv'}")
    print(f"[INFO] Saved: {output_dir / 'best_sigma_actual_only_prediction_per_clip.csv'}")


if __name__ == "__main__":
    main()
