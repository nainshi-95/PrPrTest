import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sigma_to_tag(sigma: float) -> str:
    """
    0.20 -> s020
    0.25 -> s025
    0.80 -> s080
    """
    v = int(round(sigma * 100))
    return f"s{v:03d}"


def parse_sigma_list(sigmas_str: str) -> List[float]:
    """
    Example:
        "0.20,0.25,0.30,0.35"
    """
    vals = []
    for x in sigmas_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def lower_sort_key(x: str):
    return x.lower()


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ------------------------------------------------------------
# X-axis loader
# ------------------------------------------------------------
def load_x_axis_data(
    x_dir: Path,
    x_prefix: str,
    sigma_list: List[float],
    x_metrics: List[str],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Returns:
        x_data[(clip_name, sigma_tag)] = {
            "abs_reduction": "...",
            ...
        }
    """
    x_data: Dict[Tuple[str, str], Dict[str, str]] = {}

    for sigma in sigma_list:
        sigma_tag = sigma_to_tag(sigma)
        csv_name = f"{x_prefix}_{sigma_tag}.csv"
        csv_path = x_dir / csv_name

        if not csv_path.exists():
            print(f"[WARN] X-axis CSV not found: {csv_path}")
            continue

        rows = read_csv_rows(csv_path)

        for row in rows:
            clip_name = row.get("clip_name", "").strip()
            if not clip_name:
                continue

            rec = {}
            for m in x_metrics:
                rec[m] = row.get(m, "")

            x_data[(clip_name, sigma_tag)] = rec

    return x_data


# ------------------------------------------------------------
# Y-axis loader
# ------------------------------------------------------------
def find_rd_points_csvs(folder: Path) -> List[Path]:
    return sorted(folder.rglob("rd_points.csv"))


def load_y_axis_data(
    y_root: Path,
    sigma_list: List[float],
    folder_prefix: str,
    folder_postfix: str,
    qps: List[int],
    y_metrics: List[str],
) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    """
    Returns:
        y_data[(clip_id, sigma_tag, qp)] = {
            "kbps_gt": "...",
            "kbps_blur": "...",
            ...
        }
    """
    y_data: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    qps_set = set(qps)

    for sigma in sigma_list:
        sigma_tag = sigma_to_tag(sigma)
        folder_name = f"{folder_prefix}_{sigma_tag}_{folder_postfix}"
        sigma_dir = y_root / folder_name

        if not sigma_dir.exists():
            print(f"[WARN] Y-axis sigma folder not found: {sigma_dir}")
            continue

        rd_csvs = find_rd_points_csvs(sigma_dir)
        if not rd_csvs:
            print(f"[WARN] No rd_points.csv found under: {sigma_dir}")
            continue

        for csv_path in rd_csvs:
            rows = read_csv_rows(csv_path)

            for row in rows:
                clip_id = row.get("clip_id", "").strip()
                qp_str = row.get("qp", "").strip()

                if not clip_id or not qp_str:
                    continue

                try:
                    qp = int(float(qp_str))
                except ValueError:
                    continue

                if qp not in qps_set:
                    continue

                rec = {}
                for m in y_metrics:
                    rec[m] = row.get(m, "")

                y_data[(clip_id, sigma_tag, qp)] = rec

    return y_data


# ------------------------------------------------------------
# Collect clip names
# ------------------------------------------------------------
def collect_all_clip_names(
    x_data: Dict[Tuple[str, str], Dict[str, str]],
    y_data: Dict[Tuple[str, str, int], Dict[str, str]],
) -> List[str]:
    clip_names = set()

    for clip_name, _sigma_tag in x_data.keys():
        clip_names.add(clip_name)

    for clip_id, _sigma_tag, _qp in y_data.keys():
        clip_names.add(clip_id)

    return sorted(clip_names, key=lower_sort_key)


# ------------------------------------------------------------
# Build unified rows
# ------------------------------------------------------------
def build_output_fieldnames(
    sigma_list: List[float],
    x_metrics: List[str],
    qps: List[int],
    y_metrics: List[str],
) -> List[str]:
    fieldnames = ["clip_name"]

    for sigma in sigma_list:
        sigma_tag = sigma_to_tag(sigma)

        # X-axis columns first
        for xm in x_metrics:
            fieldnames.append(f"{xm}_{sigma_tag}")

        # Then Y-axis columns for each qp
        for qp in qps:
            for ym in y_metrics:
                fieldnames.append(f"{ym}_{sigma_tag}_qp{qp}")

    return fieldnames


def build_output_rows(
    clip_names: List[str],
    sigma_list: List[float],
    x_metrics: List[str],
    qps: List[int],
    y_metrics: List[str],
    x_data: Dict[Tuple[str, str], Dict[str, str]],
    y_data: Dict[Tuple[str, str, int], Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for clip_name in clip_names:
        out_row: Dict[str, str] = {"clip_name": clip_name}

        for sigma in sigma_list:
            sigma_tag = sigma_to_tag(sigma)

            # X-axis first
            x_rec = x_data.get((clip_name, sigma_tag), {})
            for xm in x_metrics:
                out_row[f"{xm}_{sigma_tag}"] = x_rec.get(xm, "")

            # Y-axis next, for each qp
            for qp in qps:
                y_rec = y_data.get((clip_name, sigma_tag, qp), {})
                for ym in y_metrics:
                    out_row[f"{ym}_{sigma_tag}_qp{qp}"] = y_rec.get(ym, "")

        rows.append(out_row)

    return rows


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
def save_csv(output_csv: Path, fieldnames: List[str], rows: List[Dict[str, str]]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # X-axis inputs
    parser.add_argument("--x_dir", type=str, required=True, help="Folder containing temporal_info CSVs")
    parser.add_argument(
        "--x_prefix",
        type=str,
        required=True,
        help="X-axis CSV prefix, e.g. temporal_info_satd",
    )
    parser.add_argument(
        "--x_metrics",
        type=str,
        default="abs_reduction",
        help="Comma-separated X-axis metric keys to collect from X CSVs",
    )

    # Y-axis inputs
    parser.add_argument("--y_root", type=str, required=True, help="Root folder containing sigma subfolders")
    parser.add_argument("--folder_prefix", type=str, default="LargeSubset", help="Sigma folder prefix")
    parser.add_argument("--folder_postfix", type=str, default="out", help="Sigma folder postfix")
    parser.add_argument(
        "--y_metrics",
        type=str,
        default="kbps_gt,kbps_blur,psnrY_gt_enh,psnrY_blur_deblur",
        help="Comma-separated Y-axis metric keys to collect from rd_points.csv",
    )

    # Common
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
    parser.add_argument("--output_csv", type=str, required=True)

    args = parser.parse_args()

    sigma_list = parse_sigma_list(args.sigmas)
    qps = [int(x.strip()) for x in args.qps.split(",") if x.strip()]
    x_metrics = [x.strip() for x in args.x_metrics.split(",") if x.strip()]
    y_metrics = [x.strip() for x in args.y_metrics.split(",") if x.strip()]

    x_dir = Path(args.x_dir)
    y_root = Path(args.y_root)
    output_csv = Path(args.output_csv)

    # Load data
    x_data = load_x_axis_data(
        x_dir=x_dir,
        x_prefix=args.x_prefix,
        sigma_list=sigma_list,
        x_metrics=x_metrics,
    )

    y_data = load_y_axis_data(
        y_root=y_root,
        sigma_list=sigma_list,
        folder_prefix=args.folder_prefix,
        folder_postfix=args.folder_postfix,
        qps=qps,
        y_metrics=y_metrics,
    )

    # Collect clips
    clip_names = collect_all_clip_names(x_data, y_data)

    # Build output
    fieldnames = build_output_fieldnames(
        sigma_list=sigma_list,
        x_metrics=x_metrics,
        qps=qps,
        y_metrics=y_metrics,
    )

    rows = build_output_rows(
        clip_names=clip_names,
        sigma_list=sigma_list,
        x_metrics=x_metrics,
        qps=qps,
        y_metrics=y_metrics,
        x_data=x_data,
        y_data=y_data,
    )

    save_csv(output_csv, fieldnames, rows)
    print(f"[INFO] Saved unified CSV: {output_csv}")
    print(f"[INFO] Num clips: {len(rows)}")


if __name__ == "__main__":
    main()


















def safe_float(x: str):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def psnr_to_mse(psnr: float) -> float:
    return 10 ** (-psnr / 10.0)












def build_output_fieldnames(
    sigma_list: List[float],
    x_metrics: List[str],
    qps: List[int],
    y_metrics: List[str],
) -> List[str]:
    fieldnames = ["clip_name"]

    has_kbps_gt = "kbps_gt" in y_metrics
    has_kbps_blur = "kbps_blur" in y_metrics
    has_psnr_gt_enh = "psnrY_gt_enh" in y_metrics
    has_psnr_blur_deblur = "psnrY_blur_deblur" in y_metrics

    for sigma in sigma_list:
        sigma_tag = sigma_to_tag(sigma)

        # X-axis columns first
        for xm in x_metrics:
            fieldnames.append(f"{xm}_{sigma_tag}")

        # Then Y-axis columns for each qp
        for qp in qps:
            for ym in y_metrics:
                col = f"{ym}_{sigma_tag}_qp{qp}"
                fieldnames.append(col)

                # kbps_blur 바로 뒤에 delta_kbps 추가
                if ym == "kbps_blur" and has_kbps_gt:
                    fieldnames.append(f"delta_kbps_{sigma_tag}_qp{qp}")

                # psnrY_blur_deblur 바로 뒤에 delta_mse 추가
                if ym == "psnrY_blur_deblur" and has_psnr_gt_enh:
                    fieldnames.append(f"delta_mse_{sigma_tag}_qp{qp}")

    return fieldnames














def build_output_rows(
    clip_names: List[str],
    sigma_list: List[float],
    x_metrics: List[str],
    qps: List[int],
    y_metrics: List[str],
    x_data: Dict[Tuple[str, str], Dict[str, str]],
    y_data: Dict[Tuple[str, str, int], Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    has_kbps_gt = "kbps_gt" in y_metrics
    has_kbps_blur = "kbps_blur" in y_metrics
    has_psnr_gt_enh = "psnrY_gt_enh" in y_metrics
    has_psnr_blur_deblur = "psnrY_blur_deblur" in y_metrics

    for clip_name in clip_names:
        out_row: Dict[str, str] = {"clip_name": clip_name}

        for sigma in sigma_list:
            sigma_tag = sigma_to_tag(sigma)

            # X-axis first
            x_rec = x_data.get((clip_name, sigma_tag), {})
            for xm in x_metrics:
                out_row[f"{xm}_{sigma_tag}"] = x_rec.get(xm, "")

            # Y-axis next, for each qp
            for qp in qps:
                y_rec = y_data.get((clip_name, sigma_tag, qp), {})

                # 먼저 원래 y metric들 기록
                for ym in y_metrics:
                    col = f"{ym}_{sigma_tag}_qp{qp}"
                    out_row[col] = y_rec.get(ym, "")

                    # kbps_blur 바로 뒤에 delta_kbps
                    if ym == "kbps_blur" and has_kbps_gt:
                        kbps_gt = safe_float(y_rec.get("kbps_gt", ""))
                        kbps_blur = safe_float(y_rec.get("kbps_blur", ""))

                        delta_col = f"delta_kbps_{sigma_tag}_qp{qp}"
                        if kbps_gt is None or kbps_blur is None:
                            out_row[delta_col] = ""
                        else:
                            out_row[delta_col] = kbps_blur - kbps_gt

                    # psnrY_blur_deblur 바로 뒤에 delta_mse
                    if ym == "psnrY_blur_deblur" and has_psnr_gt_enh:
                        psnr_gt = safe_float(y_rec.get("psnrY_gt_enh", ""))
                        psnr_blur = safe_float(y_rec.get("psnrY_blur_deblur", ""))

                        delta_col = f"delta_mse_{sigma_tag}_qp{qp}"
                        if psnr_gt is None or psnr_blur is None:
                            out_row[delta_col] = ""
                        else:
                            mse_gt = psnr_to_mse(psnr_gt)
                            mse_blur = psnr_to_mse(psnr_blur)
                            out_row[delta_col] = mse_blur - mse_gt

        rows.append(out_row)

    return rows
