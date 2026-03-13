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


def parse_lambda_by_qp(s: str) -> Dict[int, float]:
    """
    Example:
        "22:5.623e-3,27:1.0e-2,32:2.0e-2,37:4.0e-2"
    """
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid lambda_by_qp item: {item}")
        k, v = item.split(":", 1)
        qp = int(k.strip())
        lam = float(v.strip())
        out[qp] = lam
    return out


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


# ============================================================
# Models
# ============================================================
def linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    z = np.clip(b * x, -60.0, 60.0)
    return a * np.exp(z) + c


def logistic4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    z = np.clip(-c * (x - d), -60.0, 60.0)
    return a + b / (1.0 + np.exp(z))


def poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x * x + b * x + c


def eval_model(model_name: str, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if model_name == "linear":
        return linear_model(x, *params)
    elif model_name == "exp":
        return exp_model(x, *params)
    elif model_name == "logistic":
        return logistic4(x, *params)
    elif model_name == "poly2":
        return poly2(x, *params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ============================================================
# Load regression params
# ============================================================
def load_model_bank(regression_summary_csv: Path) -> Dict[Tuple[str, int, str], Dict]:
    """
    regression_summary.csv columns expected:
      sigma_tag, qp, y_key, best_model, param_count, param_0..param_3
    """
    df = pd.read_csv(regression_summary_csv)
    bank = {}

    for _, row in df.iterrows():
        sigma_tag = str(row["sigma_tag"]).strip()
        qp = int(row["qp"])
        y_key = str(row["y_key"]).strip()
        model_name = str(row["best_model"]).strip()

        if model_name == "":
            continue

        param_count = int(row["param_count"]) if not pd.isna(row["param_count"]) else 0

        raw_params = [
            row.get("param_0", np.nan),
            row.get("param_1", np.nan),
            row.get("param_2", np.nan),
            row.get("param_3", np.nan),
        ]

        params = []
        for i in range(param_count):
            v = raw_params[i]
            if pd.isna(v):
                break
            params.append(float(v))

        if len(params) != param_count:
            continue

        bank[(sigma_tag, qp, y_key)] = {
            "model_name": model_name,
            "params": np.asarray(params, dtype=np.float64),
            "r2": float(row["best_r2"]) if not pd.isna(row["best_r2"]) else np.nan,
        }

    return bank


# ============================================================
# GT best sigma loading
# ============================================================
def load_gt_best_sigma_map(best_sigma_csv: Optional[Path], qps: List[int]) -> Dict[Tuple[str, int], float]:
    if best_sigma_csv is None:
        return {}

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
# Data lookup
# ============================================================
def get_x_value(clip_row: pd.Series, sigma: float, x_key: str):
    sigma_tag = sigma_to_tag(sigma)
    x_col = f"{x_key}_{sigma_tag}"
    return safe_float(clip_row.get(x_col, ""))


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
        "actual_delta_kbps": actual_r,
        "actual_delta_mse": actual_d,
    }


def get_predicted_values_for_sigma(
    clip_row: pd.Series,
    sigma: float,
    qp: int,
    x_key: str,
    y_rate_key: str,
    y_dist_key: str,
    model_bank: Dict[Tuple[str, int, str], Dict],
):
    sigma_tag = sigma_to_tag(sigma)

    x_val = get_x_value(clip_row, sigma, x_key)
    if x_val is None:
        return None

    key_r = (sigma_tag, qp, y_rate_key)
    key_d = (sigma_tag, qp, y_dist_key)

    if key_r not in model_bank or key_d not in model_bank:
        return None

    model_r = model_bank[key_r]
    model_d = model_bank[key_d]

    x_arr = np.array([x_val], dtype=np.float64)

    pred_r = float(eval_model(model_r["model_name"], x_arr, model_r["params"])[0])
    pred_d = float(eval_model(model_d["model_name"], x_arr, model_d["params"])[0])

    return {
        "x": x_val,
        "pred_delta_kbps": pred_r,
        "pred_delta_mse": pred_d,
    }


# ============================================================
# Metrics
# ============================================================
def r2_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot < 1e-18:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# ============================================================
# Fixed-lambda evaluation
# ============================================================
def evaluate_fixed_lambda_for_qp(
    unified_df: pd.DataFrame,
    clip_col: str,
    sigma_list: List[float],
    qp: int,
    lam: float,
    x_key: str,
    y_rate_key: str,
    y_dist_key: str,
    model_bank: Dict[Tuple[str, int, str], Dict],
    gt_best_sigma_map: Dict[Tuple[str, int], float],
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Returns:
      summary_row
      per_clip_selector_rows
      per_sigma_point_rows

    per_sigma_point_rows:
      one row per (clip, sigma, qp), with actual/predicted delta and J errors

    per_clip_selector_rows:
      one row per clip, with predicted-best sigma, actual-best sigma, regret
    """
    step = sigma_step(sigma_list)

    # pointwise J prediction quality
    all_actual_J = []
    all_pred_J = []
    all_J_err = []
    all_abs_J_err = []

    # selector quality
    sigma_errors_vs_gtcsv = []
    soft_scores_vs_gtcsv = []
    exact_hits = 0
    one_step_hits = 0
    two_step_hits = 0

    regrets = []
    normalized_regrets = []

    per_sigma_point_rows = []
    per_clip_selector_rows = []

    for _, row in unified_df.iterrows():
        clip_name = str(row[clip_col]).strip()
        if not clip_name:
            continue

        candidate_rows = []

        for sigma in sigma_list:
            sigma_tag = sigma_to_tag(sigma)

            pred = get_predicted_values_for_sigma(
                clip_row=row,
                sigma=sigma,
                qp=qp,
                x_key=x_key,
                y_rate_key=y_rate_key,
                y_dist_key=y_dist_key,
                model_bank=model_bank,
            )
            actual = get_actual_values_for_sigma(
                clip_row=row,
                sigma=sigma,
                qp=qp,
                y_rate_key=y_rate_key,
                y_dist_key=y_dist_key,
            )

            if pred is None or actual is None:
                continue

            pred_J = pred["pred_delta_mse"] + lam * pred["pred_delta_kbps"]
            actual_J = actual["actual_delta_mse"] + lam * actual["actual_delta_kbps"]

            J_err = pred_J - actual_J
            abs_J_err = abs(J_err)

            all_actual_J.append(actual_J)
            all_pred_J.append(pred_J)
            all_J_err.append(J_err)
            all_abs_J_err.append(abs_J_err)

            point_row = {
                "clip_name": clip_name,
                "qp": qp,
                "lambda": lam,
                "sigma": sigma,
                "sigma_tag": sigma_tag,
                "x": pred["x"],

                "actual_delta_kbps": actual["actual_delta_kbps"],
                "pred_delta_kbps": pred["pred_delta_kbps"],
                "error_delta_kbps": pred["pred_delta_kbps"] - actual["actual_delta_kbps"],

                "actual_delta_mse": actual["actual_delta_mse"],
                "pred_delta_mse": pred["pred_delta_mse"],
                "error_delta_mse": pred["pred_delta_mse"] - actual["actual_delta_mse"],

                "actual_J": actual_J,
                "pred_J": pred_J,
                "error_J": J_err,
                "abs_error_J": abs_J_err,
            }
            per_sigma_point_rows.append(point_row)

            candidate_rows.append({
                "sigma": sigma,
                "sigma_tag": sigma_tag,
                "pred_J": pred_J,
                "actual_J": actual_J,
                "actual_delta_kbps": actual["actual_delta_kbps"],
                "actual_delta_mse": actual["actual_delta_mse"],
                "pred_delta_kbps": pred["pred_delta_kbps"],
                "pred_delta_mse": pred["pred_delta_mse"],
            })

        if not candidate_rows:
            continue

        pred_best = min(candidate_rows, key=lambda z: z["pred_J"])
        actual_best = min(candidate_rows, key=lambda z: z["actual_J"])

        pred_best_sigma = float(pred_best["sigma"])
        actual_best_sigma = float(actual_best["sigma"])

        regret = pred_best["actual_J"] - actual_best["actual_J"]
        normalized_regret = regret / (abs(actual_best["actual_J"]) + 1e-12)

        regrets.append(regret)
        normalized_regrets.append(normalized_regret)

        gt_sigma_csv = gt_best_sigma_map.get((clip_name, qp), None)
        sigma_error_csv = None
        soft_score_csv = None

        if gt_sigma_csv is not None:
            sigma_error_csv = abs(pred_best_sigma - gt_sigma_csv)
            sigma_errors_vs_gtcsv.append(sigma_error_csv)

            soft_score_csv = math.exp(-sigma_error_csv / max(step, 1e-12))
            soft_scores_vs_gtcsv.append(soft_score_csv)

            if sigma_error_csv < 1e-12:
                exact_hits += 1
            if sigma_error_csv <= step + 1e-12:
                one_step_hits += 1
            if sigma_error_csv <= 2.0 * step + 1e-12:
                two_step_hits += 1

        per_clip_selector_rows.append({
            "clip_name": clip_name,
            "qp": qp,
            "lambda": lam,

            "pred_best_sigma": pred_best_sigma,
            "actual_best_sigma_from_actualJ": actual_best_sigma,

            "pred_best_sigma_tag": pred_best["sigma_tag"],
            "actual_best_sigma_tag": actual_best["sigma_tag"],

            "pred_best_actual_J": pred_best["actual_J"],
            "actual_best_actual_J": actual_best["actual_J"],

            "pred_best_actual_delta_kbps": pred_best["actual_delta_kbps"],
            "pred_best_actual_delta_mse": pred_best["actual_delta_mse"],

            "actual_best_actual_delta_kbps": actual_best["actual_delta_kbps"],
            "actual_best_actual_delta_mse": actual_best["actual_delta_mse"],

            "regret": regret,
            "normalized_regret": normalized_regret,

            "gt_best_sigma_csv": gt_sigma_csv if gt_sigma_csv is not None else "",
            "abs_sigma_error_vs_gtcsv": sigma_error_csv if sigma_error_csv is not None else "",
            "soft_score_vs_gtcsv": soft_score_csv if soft_score_csv is not None else "",
        })

    # summary
    summary = {
        "qp": qp,
        "lambda": lam,
        "num_sigma_points": len(all_actual_J),
        "num_clips": len(per_clip_selector_rows),
    }

    if len(all_actual_J) > 0:
        actual_J_arr = np.asarray(all_actual_J, dtype=np.float64)
        pred_J_arr = np.asarray(all_pred_J, dtype=np.float64)
        J_err_arr = np.asarray(all_J_err, dtype=np.float64)
        abs_J_err_arr = np.asarray(all_abs_J_err, dtype=np.float64)

        summary.update({
            "mae_J": float(np.mean(abs_J_err_arr)),
            "rmse_J": float(np.sqrt(np.mean(J_err_arr ** 2))),
            "bias_J": float(np.mean(J_err_arr)),
            "r2_J": r2_from_arrays(actual_J_arr, pred_J_arr),
        })
    else:
        summary.update({
            "mae_J": np.nan,
            "rmse_J": np.nan,
            "bias_J": np.nan,
            "r2_J": np.nan,
        })

    if len(regrets) > 0:
        regrets_arr = np.asarray(regrets, dtype=np.float64)
        norm_regrets_arr = np.asarray(normalized_regrets, dtype=np.float64)
        summary.update({
            "mean_regret": float(np.mean(regrets_arr)),
            "median_regret": float(np.median(regrets_arr)),
            "mean_normalized_regret": float(np.mean(norm_regrets_arr)),
            "median_normalized_regret": float(np.median(norm_regrets_arr)),
        })
    else:
        summary.update({
            "mean_regret": np.nan,
            "median_regret": np.nan,
            "mean_normalized_regret": np.nan,
            "median_normalized_regret": np.nan,
        })

    if len(sigma_errors_vs_gtcsv) > 0:
        sigma_err_arr = np.asarray(sigma_errors_vs_gtcsv, dtype=np.float64)
        soft_arr = np.asarray(soft_scores_vs_gtcsv, dtype=np.float64)
        n = len(sigma_err_arr)

        summary.update({
            "mae_sigma_vs_gtcsv": float(np.mean(sigma_err_arr)),
            "rmse_sigma_vs_gtcsv": float(np.sqrt(np.mean(sigma_err_arr ** 2))),
            "mean_soft_score_vs_gtcsv": float(np.mean(soft_arr)),
            "exact_acc_vs_gtcsv": float(exact_hits / n),
            "within_1step_acc_vs_gtcsv": float(one_step_hits / n),
            "within_2step_acc_vs_gtcsv": float(two_step_hits / n),
        })
    else:
        summary.update({
            "mae_sigma_vs_gtcsv": np.nan,
            "rmse_sigma_vs_gtcsv": np.nan,
            "mean_soft_score_vs_gtcsv": np.nan,
            "exact_acc_vs_gtcsv": np.nan,
            "within_1step_acc_vs_gtcsv": np.nan,
            "within_2step_acc_vs_gtcsv": np.nan,
        })

    return summary, per_clip_selector_rows, per_sigma_point_rows


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unified_csv", type=str, required=True)
    parser.add_argument("--regression_summary_csv", type=str, required=True)
    parser.add_argument("--best_sigma_csv", type=str, default=None)

    parser.add_argument(
        "--sigmas",
        type=str,
        required=True,
        help='e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"',
    )
    parser.add_argument("--qps", type=str, default="22,27,32,37")

    parser.add_argument("--lambda_by_qp", type=str, required=True)
    parser.add_argument("--clip_col", type=str, default="clip_name")
    parser.add_argument("--x_key", type=str, default="abs_reduction")
    parser.add_argument("--y_rate_key", type=str, default="delta_kbps")
    parser.add_argument("--y_dist_key", type=str, default="delta_mse")

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    unified_df = pd.read_csv(args.unified_csv)
    model_bank = load_model_bank(Path(args.regression_summary_csv))

    sigma_list = parse_sigma_list(args.sigmas)
    qps = parse_qp_list(args.qps)
    lambda_by_qp = parse_lambda_by_qp(args.lambda_by_qp)

    gt_best_sigma_map = load_gt_best_sigma_map(Path(args.best_sigma_csv), qps) if args.best_sigma_csv else {}

    summary_rows = []
    all_selector_rows = []
    all_sigma_point_rows = []

    for qp in qps:
        if qp not in lambda_by_qp:
            print(f"[WARN] No lambda provided for qp={qp}, skip")
            continue

        lam = float(lambda_by_qp[qp])

        print(f"[INFO] Evaluating fixed lambda for qp={qp}: lambda={lam:.6e}")

        summary, selector_rows, sigma_point_rows = evaluate_fixed_lambda_for_qp(
            unified_df=unified_df,
            clip_col=args.clip_col,
            sigma_list=sigma_list,
            qp=qp,
            lam=lam,
            x_key=args.x_key,
            y_rate_key=args.y_rate_key,
            y_dist_key=args.y_dist_key,
            model_bank=model_bank,
            gt_best_sigma_map=gt_best_sigma_map,
        )

        summary_rows.append(summary)
        all_selector_rows.extend(selector_rows)
        all_sigma_point_rows.extend(sigma_point_rows)

    pd.DataFrame(summary_rows).to_csv(output_dir / "fixed_lambda_summary.csv", index=False)
    pd.DataFrame(all_selector_rows).to_csv(output_dir / "fixed_lambda_selector_per_clip.csv", index=False)
    pd.DataFrame(all_sigma_point_rows).to_csv(output_dir / "fixed_lambda_J_pointwise.csv", index=False)

    print(f"[INFO] Saved: {output_dir / 'fixed_lambda_summary.csv'}")
    print(f"[INFO] Saved: {output_dir / 'fixed_lambda_selector_per_clip.csv'}")
    print(f"[INFO] Saved: {output_dir / 'fixed_lambda_J_pointwise.csv'}")


if __name__ == "__main__":
    main()
