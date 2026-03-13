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
# Regression summary loading
# ============================================================
def load_model_bank(regression_summary_csv: Path) -> Dict[Tuple[str, int, str], Dict]:
    """
    Returns:
        model_bank[(sigma_tag, qp, y_key)] = {
            "model_name": str,
            "params": np.ndarray,
            "r2": float,
        }
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
# Ground-truth best sigma loading
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
    out = {}

    if "clip_id" not in df.columns:
        raise KeyError("best sigma CSV must contain 'clip_id' column")

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
# Prediction helpers
# ============================================================
def predict_delta_for_clip_sigma_qp(
    clip_row: pd.Series,
    sigma: float,
    qp: int,
    x_key: str,
    y_rate_key: str,
    y_dist_key: str,
    model_bank: Dict[Tuple[str, int, str], Dict],
) -> Optional[Dict]:
    sigma_tag = sigma_to_tag(sigma)
    x_col = f"{x_key}_{sigma_tag}"

    x_val = safe_float(clip_row.get(x_col, ""))
    if x_val is None:
        return None

    key_r = (sigma_tag, qp, y_rate_key)
    key_d = (sigma_tag, qp, y_dist_key)

    if key_r not in model_bank or key_d not in model_bank:
        return None

    mr = model_bank[key_r]
    md = model_bank[key_d]

    x_arr = np.array([x_val], dtype=np.float64)
    pred_r = float(eval_model(mr["model_name"], x_arr, mr["params"])[0])
    pred_d = float(eval_model(md["model_name"], x_arr, md["params"])[0])

    return {
        "sigma": sigma,
        "sigma_tag": sigma_tag,
        "x": x_val,
        "pred_delta_kbps": pred_r,
        "pred_delta_mse": pred_d,
    }


# ============================================================
# Lambda sweep
# ============================================================
def build_lambda_candidates(min_log10: float, max_log10: float, num: int) -> np.ndarray:
    return np.logspace(min_log10, max_log10, num=num)


def evaluate_lambda_for_qp(
    unified_df: pd.DataFrame,
    clip_col: str,
    sigma_list: List[float],
    qp: int,
    x_key: str,
    y_rate_key: str,
    y_dist_key: str,
    model_bank: Dict[Tuple[str, int, str], Dict],
    gt_best_sigma_map: Dict[Tuple[str, int], float],
    lam: float,
) -> Tuple[Dict, List[Dict]]:
    """
    For one qp and one lambda:
      - predict best sigma per clip by minimizing J
      - compare to GT best sigma
    """
    step = sigma_step(sigma_list)

    errors = []
    soft_scores = []
    exact_hits = 0
    one_step_hits = 0
    two_step_hits = 0
    total = 0

    per_clip_rows = []

    for _, row in unified_df.iterrows():
        clip_name = str(row[clip_col]).strip()
        gt_sigma = gt_best_sigma_map.get((clip_name, qp), None)
        if gt_sigma is None:
            continue

        candidates = []
        for sigma in sigma_list:
            pred = predict_delta_for_clip_sigma_qp(
                clip_row=row,
                sigma=sigma,
                qp=qp,
                x_key=x_key,
                y_rate_key=y_rate_key,
                y_dist_key=y_dist_key,
                model_bank=model_bank,
            )
            if pred is None:
                continue

            J = pred["pred_delta_mse"] + lam * pred["pred_delta_kbps"]
            pred["J"] = J
            candidates.append(pred)

        if not candidates:
            continue

        best_pred = min(candidates, key=lambda z: z["J"])
        pred_sigma = float(best_pred["sigma"])

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

        per_clip_rows.append({
            "clip_name": clip_name,
            "qp": qp,
            "lambda": lam,
            "gt_best_sigma": gt_sigma,
            "pred_best_sigma": pred_sigma,
            "abs_sigma_error": err,
            "soft_score": soft_score,
            "chosen_sigma_tag": best_pred["sigma_tag"],
            "x": best_pred["x"],
            "pred_delta_kbps": best_pred["pred_delta_kbps"],
            "pred_delta_mse": best_pred["pred_delta_mse"],
            "J": best_pred["J"],
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

    return summary, per_clip_rows


def select_best_lambda(lambda_rows: List[Dict]) -> Optional[Dict]:
    """
    Priority:
      1) smallest MAE
      2) largest mean_soft_score
      3) largest within_1step_acc
      4) largest within_2step_acc
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
    parser.add_argument("--regression_summary_csv", type=str, required=True)
    parser.add_argument("--best_sigma_csv", type=str, required=True)

    parser.add_argument(
        "--sigmas",
        type=str,
        required=True,
        help='e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"',
    )
    parser.add_argument("--qps", type=str, default="22,27,32,37")

    parser.add_argument("--clip_col", type=str, default="clip_name")
    parser.add_argument("--x_key", type=str, default="abs_reduction")
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
    model_bank = load_model_bank(Path(args.regression_summary_csv))

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
        print(f"[INFO] Sweeping lambda for qp={qp} ...")

        qp_lambda_rows = []
        for lam in lambda_candidates:
            summary, _ = evaluate_lambda_for_qp(
                unified_df=unified_df,
                clip_col=args.clip_col,
                sigma_list=sigma_list,
                qp=qp,
                x_key=args.x_key,
                y_rate_key=args.y_rate_key,
                y_dist_key=args.y_dist_key,
                model_bank=model_bank,
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
            f"[INFO] Best lambda for qp={qp}: {best_lambda['lambda']:.6e}, "
            f"MAE={best_lambda['mae_sigma']:.6f}, "
            f"soft={best_lambda['mean_soft_score']:.6f}, "
            f"1step={best_lambda['within_1step_acc']:.6f}, "
            f"2step={best_lambda['within_2step_acc']:.6f}"
        )

        # save qp-specific sweep
        pd.DataFrame(qp_lambda_rows).to_csv(output_dir / f"lambda_sweep_qp{qp}.csv", index=False)

        # evaluate once more with selected lambda to save per-clip predictions
        _, per_clip_rows = evaluate_lambda_for_qp(
            unified_df=unified_df,
            clip_col=args.clip_col,
            sigma_list=sigma_list,
            qp=qp,
            x_key=args.x_key,
            y_rate_key=args.y_rate_key,
            y_dist_key=args.y_dist_key,
            model_bank=model_bank,
            gt_best_sigma_map=gt_best_sigma_map,
            lam=float(best_lambda["lambda"]),
        )
        best_pred_rows.extend(per_clip_rows)

    # save outputs
    pd.DataFrame(all_lambda_rows).to_csv(output_dir / "lambda_sweep_all.csv", index=False)
    pd.DataFrame(best_lambda_rows).to_csv(output_dir / "best_lambda_per_qp.csv", index=False)
    pd.DataFrame(best_pred_rows).to_csv(output_dir / "best_sigma_prediction_per_clip.csv", index=False)

    print(f"[INFO] Saved: {output_dir / 'lambda_sweep_all.csv'}")
    print(f"[INFO] Saved: {output_dir / 'best_lambda_per_qp.csv'}")
    print(f"[INFO] Saved: {output_dir / 'best_sigma_prediction_per_clip.csv'}")


if __name__ == "__main__":
    main()
