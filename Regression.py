import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# ============================================================
# Utilities
# ============================================================
def sigma_to_tag(sigma: float) -> str:
    """
    0.20 -> s020
    0.25 -> s025
    0.80 -> s080
    """
    v = int(round(sigma * 100))
    return f"s{v:03d}"


def parse_sigma_list(sigmas_str: str) -> List[float]:
    vals = []
    for x in sigmas_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def parse_qp_list(qps_str: str) -> List[int]:
    vals = []
    for x in qps_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ============================================================
# Models
# ============================================================
def linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    y = a * exp(b*x) + c
    overflow 방지를 위해 exponent clip
    """
    z = np.clip(b * x, -60.0, 60.0)
    return a * np.exp(z) + c


def logistic4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    y = a + b / (1 + exp(-c*(x-d)))
    overflow 방지를 위해 exponent clip
    """
    z = np.clip(-c * (x - d), -60.0, 60.0)
    return a + b / (1.0 + np.exp(z))


def poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x * x + b * x + c


# ============================================================
# Fitting
# ============================================================
def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = np.polyfit(x, y, deg=1)  # [a, b]
    y_pred = np.polyval(coeffs, x)
    return coeffs, y_pred


def fit_poly2(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = np.polyfit(x, y, deg=2)  # [a, b, c]
    y_pred = np.polyval(coeffs, x)
    return coeffs, y_pred


def fit_exp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = float(np.max(x) - np.min(x))
    if x_span < 1e-12:
        x_span = 1.0

    # 초기값
    p0 = np.array([
        max(y_max - y_min, 1e-6),   # a
        1.0 / x_span,               # b
        y_min,                      # c
    ], dtype=np.float64)

    lower = np.array([-np.inf, -100.0, -np.inf], dtype=np.float64)
    upper = np.array([ np.inf,  100.0,  np.inf], dtype=np.float64)

    params, _ = curve_fit(
        exp_model,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=200000,
    )

    y_pred = exp_model(x, *params)
    return params, y_pred


def fit_logistic4(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0

    p0 = np.array([
        y_min,                  # a
        y_max - y_min,          # b
        1.0 / x_std,            # c
        x_mean,                 # d
    ], dtype=np.float64)

    lower = np.array([-np.inf, -np.inf, -100.0, -np.inf], dtype=np.float64)
    upper = np.array([ np.inf,  np.inf,  100.0,  np.inf], dtype=np.float64)

    params, _ = curve_fit(
        logistic4,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=200000,
    )

    y_pred = logistic4(x, *params)
    return params, y_pred


def try_fit_model(model_name: str, x: np.ndarray, y: np.ndarray):
    if model_name == "linear":
        params, y_pred = fit_linear(x, y)
    elif model_name == "poly2":
        params, y_pred = fit_poly2(x, y)
    elif model_name == "exp":
        params, y_pred = fit_exp(x, y)
    elif model_name == "logistic":
        params, y_pred = fit_logistic4(x, y)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    r2 = r2_score(y, y_pred)
    return params, y_pred, r2


# ============================================================
# Equation formatting
# ============================================================
def format_equation(model_name: str, params: np.ndarray) -> str:
    if model_name == "linear":
        a, b = params
        return f"y = {a:.6g} x + {b:.6g}"
    elif model_name == "poly2":
        a, b, c = params
        return f"y = {a:.6g} x^2 + {b:.6g} x + {c:.6g}"
    elif model_name == "exp":
        a, b, c = params
        return f"y = {a:.6g} exp({b:.6g} x) + {c:.6g}"
    elif model_name == "logistic":
        a, b, c, d = params
        return f"y = {a:.6g} + {b:.6g}/(1 + exp(-{c:.6g}(x - {d:.6g})))"
    else:
        return "Unknown model"


# ============================================================
# Plotting
# ============================================================
def eval_model(model_name: str, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if model_name == "linear":
        return linear_model(x, *params)
    elif model_name == "poly2":
        return poly2(x, *params)
    elif model_name == "exp":
        return exp_model(x, *params)
    elif model_name == "logistic":
        return logistic4(x, *params)
    else:
        raise ValueError(model_name)


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    params: np.ndarray,
    r2: float,
    title: str,
    x_label: str,
    y_label: str,
    save_path: Path,
):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x_max - x_min) < 1e-12:
        x_plot = np.array([x_min - 1e-6, x_max + 1e-6], dtype=np.float64)
    else:
        pad = 0.05 * (x_max - x_min)
        x_plot = np.linspace(x_min - pad, x_max + pad, 400)

    y_plot = eval_model(model_name, x_plot, params)
    plt.plot(x_plot, y_plot)

    eq = format_equation(model_name, params)
    text = f"{eq}\nR^2 = {r2:.6f}"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(
        0.03,
        0.97,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Core regression routine
# ============================================================
def build_xy_from_dataframe(
    df: pd.DataFrame,
    sigma_tag: str,
    qp: int,
    x_key_base: str,
    y_key_base: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_col = f"{x_key_base}_{sigma_tag}"
    y_col = f"{y_key_base}_{sigma_tag}_qp{qp}"

    if x_col not in df.columns:
        raise KeyError(f"Missing X column: {x_col}")
    if y_col not in df.columns:
        raise KeyError(f"Missing Y column: {y_col}")

    x = safe_numeric_series(df[x_col])
    y = safe_numeric_series(df[y_col])

    valid = ~(x.isna() | y.isna())
    x = x[valid].to_numpy(dtype=np.float64)
    y = y[valid].to_numpy(dtype=np.float64)

    return x, y


def fit_one_pair_best_model(
    df: pd.DataFrame,
    sigma_tag: str,
    qp: int,
    x_key_base: str,
    y_key_base: str,
    candidate_models: List[str],
    min_points: int = 4,
):
    x, y = build_xy_from_dataframe(df, sigma_tag, qp, x_key_base, y_key_base)

    result = {
        "sigma_tag": sigma_tag,
        "qp": qp,
        "x_key": x_key_base,
        "y_key": y_key_base,
        "num_points": int(len(x)),
        "best_model": "",
        "best_r2": np.nan,
        "equation": "",
        "params": None,
        "x": x,
        "y": y,
    }

    if len(x) < min_points:
        return result

    best = None

    for model_name in candidate_models:
        try:
            params, _y_pred, r2 = try_fit_model(model_name, x, y)
            if np.isnan(r2):
                continue
            if (best is None) or (r2 > best["r2"]):
                best = {
                    "model_name": model_name,
                    "params": params,
                    "r2": r2,
                }
        except Exception as e:
            print(f"[WARN] Fit failed: sigma={sigma_tag}, qp={qp}, y={y_key_base}, model={model_name}, err={e}")

    if best is None:
        return result

    result["best_model"] = best["model_name"]
    result["best_r2"] = float(best["r2"])
    result["equation"] = format_equation(best["model_name"], best["params"])
    result["params"] = best["params"]

    return result


# ============================================================
# Save summary csv
# ============================================================
def save_regression_summary_csv(results: List[Dict], output_csv: Path):
    ensure_dir(output_csv.parent)

    rows = []
    for r in results:
        rows.append({
            "sigma_tag": r["sigma_tag"],
            "qp": r["qp"],
            "x_key": r["x_key"],
            "y_key": r["y_key"],
            "num_points": r["num_points"],
            "best_model": r["best_model"],
            "best_r2": r["best_r2"],
            "equation": r["equation"],
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Unified summary CSV")
    parser.add_argument(
        "--sigmas",
        type=str,
        required=True,
        help='Comma-separated sigma list, e.g. "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"',
    )
    parser.add_argument(
        "--qps",
        type=str,
        default="22,27,32,37",
        help='Comma-separated qp list, e.g. "22,27,32,37"',
    )
    parser.add_argument(
        "--x_key",
        type=str,
        default="abs_reduction",
        help="Base X-axis key, e.g. abs_reduction",
    )
    parser.add_argument(
        "--y1_key",
        type=str,
        default="delta_kbps",
        help="Base Y1-axis key, e.g. delta_kbps",
    )
    parser.add_argument(
        "--y2_key",
        type=str,
        default="delta_mse",
        help="Base Y2-axis key, e.g. delta_mse",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="linear,exp,logistic,poly2",
        help='Candidate models, comma-separated. Example: "linear,exp,logistic,poly2"',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots and summary CSV",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    figs_dir = output_dir / "figures"
    ensure_dir(figs_dir)

    sigma_list = parse_sigma_list(args.sigmas)
    sigma_tags = [sigma_to_tag(s) for s in sigma_list]
    qps = parse_qp_list(args.qps)
    candidate_models = [m.strip() for m in args.models.split(",") if m.strip()]

    df = pd.read_csv(input_csv)

    all_results: List[Dict] = []

    target_specs = [
        ("y1", args.y1_key, "Delta kbps"),
        ("y2", args.y2_key, "Delta MSE"),
    ]

    for sigma_tag in sigma_tags:
        for qp in qps:
            for _target_name, y_key_base, y_label in target_specs:
                try:
                    result = fit_one_pair_best_model(
                        df=df,
                        sigma_tag=sigma_tag,
                        qp=qp,
                        x_key_base=args.x_key,
                        y_key_base=y_key_base,
                        candidate_models=candidate_models,
                    )
                except KeyError as e:
                    print(f"[WARN] Skip: {e}")
                    continue

                all_results.append(result)

                if result["best_model"] == "":
                    print(
                        f"[WARN] No valid fit for sigma={sigma_tag}, qp={qp}, y={y_key_base}, "
                        f"num_points={result['num_points']}"
                    )
                    continue

                fig_name = sanitize_filename(f"{y_key_base}_{sigma_tag}_qp{qp}_{result['best_model']}.png")
                fig_path = figs_dir / fig_name

                plot_fit(
                    x=result["x"],
                    y=result["y"],
                    model_name=result["best_model"],
                    params=result["params"],
                    r2=result["best_r2"],
                    title=f"{y_key_base} vs {args.x_key} ({sigma_tag}, qp={qp})",
                    x_label=args.x_key,
                    y_label=y_label,
                    save_path=fig_path,
                )

                print(
                    f"[INFO] sigma={sigma_tag}, qp={qp}, y={y_key_base}, "
                    f"best_model={result['best_model']}, R^2={result['best_r2']:.6f}"
                )

    summary_csv = output_dir / "regression_summary.csv"
    save_regression_summary_csv(all_results, summary_csv)
    print(f"[INFO] Saved summary CSV: {summary_csv}")
    print(f"[INFO] Saved figures dir: {figs_dir}")


if __name__ == "__main__":
    main()
