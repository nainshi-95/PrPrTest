#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sigma_tags(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty --sigmas")
    return vals


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# Column helpers
# ============================================================
def col_delta_kbps(sigma_tag: str, qp: int) -> str:
    return f"delta_kbps_{sigma_tag}_qp{qp}"


def col_psnr_like(sigma_tag: str, qp: int) -> str:
    # 실제 이름은 psnrY_blur_deblur인데
    # 내부 값은 mse로 저장돼 있어도 이 key로 접근
    return f"psnrY_blur_deblur_{sigma_tag}_qp{qp}"


def sigma_tag_to_value(tag: str) -> float:
    # s020 -> 0.20
    if not tag.startswith("s"):
        raise ValueError(f"Bad sigma tag: {tag}")
    return float(int(tag[1:])) / 100.0


# ============================================================
# Data prep
# ============================================================
def load_feature_csv(feature_csv: Path, clip_col: str) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    df.columns = [c.strip() for c in df.columns]
    if clip_col not in df.columns:
        raise KeyError(f"Missing clip column in feature csv: {clip_col}")
    return df


def load_target_csv(target_csv: Path, clip_col: str) -> pd.DataFrame:
    df = pd.read_csv(target_csv)
    df.columns = [c.strip() for c in df.columns]
    if clip_col not in df.columns:
        raise KeyError(f"Missing clip column in target csv: {clip_col}")
    return df


def build_long_dataframe(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    clip_col: str,
    qp: int,
    sigma_tags: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    feature_df: one row per clip
    target_df : one row per clip, wide columns
    output    : one row per (clip, sigma)
    """
    merged = pd.merge(feature_df, target_df, on=clip_col, how="inner")
    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Check clip names.")

    feature_cols = [c for c in feature_df.columns if c != clip_col]
    rows = []

    for _, row in merged.iterrows():
        clip_name = row[clip_col]

        for sigma_tag in sigma_tags:
            c_r = col_delta_kbps(sigma_tag, qp)
            c_d = col_psnr_like(sigma_tag, qp)

            if c_r not in merged.columns:
                continue
            if c_d not in merged.columns:
                continue

            r = safe_float(row[c_r])
            d = safe_float(row[c_d])

            if not np.isfinite(r) or not np.isfinite(d):
                continue

            item = {
                clip_col: clip_name,
                "sigma_tag": sigma_tag,
                "sigma_value": sigma_tag_to_value(sigma_tag),
                "target_r": float(r),
                "target_d": float(d),
            }
            for fc in feature_cols:
                item[fc] = safe_float(row[fc])

            rows.append(item)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No valid training rows built from feature/target csv.")

    # feature numeric clean
    valid_mask = np.ones(len(long_df), dtype=bool)
    for fc in feature_cols:
        v = pd.to_numeric(long_df[fc], errors="coerce").to_numpy(np.float64)
        valid_mask &= np.isfinite(v)

    valid_mask &= np.isfinite(long_df["sigma_value"].to_numpy(np.float64))
    valid_mask &= np.isfinite(long_df["target_r"].to_numpy(np.float64))
    valid_mask &= np.isfinite(long_df["target_d"].to_numpy(np.float64))

    long_df = long_df.loc[valid_mask].reset_index(drop=True)
    if long_df.empty:
        raise RuntimeError("All rows became invalid after numeric filtering.")

    return long_df, feature_cols


def split_by_clip(
    long_df: pd.DataFrame,
    clip_col: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clips = sorted(long_df[clip_col].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(clips)

    n_train = max(1, int(round(len(clips) * train_ratio)))
    n_train = min(n_train, len(clips) - 1) if len(clips) >= 2 else len(clips)

    train_clips = set(clips[:n_train])
    val_clips = set(clips[n_train:])

    train_df = long_df[long_df[clip_col].isin(train_clips)].reset_index(drop=True)
    val_df = long_df[long_df[clip_col].isin(val_clips)].reset_index(drop=True)

    if len(val_df) == 0:
        val_df = train_df.iloc[:0].copy()

    return train_df, val_df


# ============================================================
# Normalizer
# ============================================================
class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def load_state_dict(self, d):
        self.mean = np.asarray(d["mean"], dtype=np.float32)
        self.std = np.asarray(d["std"], dtype=np.float32)


# ============================================================
# Dataset
# ============================================================
class SigmaRDRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        clip_col: str,
        x_std: Standardizer | None = None,
        y_std: Standardizer | None = None,
        fit_x: bool = False,
        fit_y: bool = False,
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.clip_col = clip_col

        x_feat = self.df[feature_cols].to_numpy(np.float32)
        sigma = self.df[["sigma_value"]].to_numpy(np.float32)
        self.x = np.concatenate([x_feat, sigma], axis=1).astype(np.float32)

        self.y = self.df[["target_r", "target_d"]].to_numpy(np.float32)

        self.x_std = x_std if x_std is not None else Standardizer()
        self.y_std = y_std if y_std is not None else Standardizer()

        if fit_x:
            self.x_std.fit(self.x)
        if fit_y:
            self.y_std.fit(self.y)

        self.xn = self.x_std.transform(self.x).astype(np.float32)
        self.yn = self.y_std.transform(self.y).astype(np.float32)

        self.clip_names = self.df[clip_col].tolist()
        self.sigma_tags = self.df["sigma_tag"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.xn[idx]),
            "y": torch.from_numpy(self.yn[idx]),
            "clip_name": self.clip_names[idx],
            "sigma_tag": self.sigma_tags[idx],
            "target_r": torch.tensor(self.y[idx, 0], dtype=torch.float32),
            "target_d": torch.tensor(self.y[idx, 1], dtype=torch.float32),
        }


# ============================================================
# Model
# ============================================================
class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()

        layers = []
        cur = in_dim
        for i in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim

        layers.append(nn.Linear(cur, 2))  # [R, D]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        loss = torch.mean((pred - y) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def predict_dataset(model, dataset: SigmaRDRegressionDataset, device) -> pd.DataFrame:
    model.eval()

    x = torch.from_numpy(dataset.xn).to(device)
    pred_n = model(x).cpu().numpy()
    pred = pred_n * dataset.y_std.std + dataset.y_std.mean

    out = dataset.df.copy()
    out["pred_r"] = pred[:, 0]
    out["pred_d"] = pred[:, 1]
    out["abs_err_r"] = np.abs(out["pred_r"] - out["target_r"])
    out["abs_err_d"] = np.abs(out["pred_d"] - out["target_d"])
    return out


def summarize_predictions(df_pred: pd.DataFrame) -> Dict[str, float]:
    yt_r = df_pred["target_r"].to_numpy(np.float64)
    yp_r = df_pred["pred_r"].to_numpy(np.float64)
    yt_d = df_pred["target_d"].to_numpy(np.float64)
    yp_d = df_pred["pred_d"].to_numpy(np.float64)

    return {
        "mae_r": mae_np(yt_r, yp_r),
        "rmse_r": rmse_np(yt_r, yp_r),
        "r2_r": r2_score_np(yt_r, yp_r),
        "mae_d": mae_np(yt_d, yp_d),
        "rmse_d": rmse_np(yt_d, yp_d),
        "r2_d": r2_score_np(yt_d, yp_d),
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_csv", type=str, required=True, help="feature csv")
    ap.add_argument("--target_csv", type=str, required=True, help="wide target csv containing delta_kbps_* and psnrY_blur_deblur_*")
    ap.add_argument("--qp", type=int, required=True, help="train one qp at a time")
    ap.add_argument("--sigmas", type=str, required=True, help='e.g. "s020,s025,s030,s035,s040"')

    ap.add_argument("--clip_col", type=str, default="clip_name")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--hidden_dim", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_tags = parse_sigma_tags(args.sigmas)

    feature_df = load_feature_csv(Path(args.dataset_csv), args.clip_col)
    target_df = load_target_csv(Path(args.target_csv), args.clip_col)

    long_df, feature_cols = build_long_dataframe(
        feature_df=feature_df,
        target_df=target_df,
        clip_col=args.clip_col,
        qp=args.qp,
        sigma_tags=sigma_tags,
    )

    train_df, val_df = split_by_clip(
        long_df=long_df,
        clip_col=args.clip_col,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    x_std = Standardizer()
    y_std = Standardizer()

    train_ds = SigmaRDRegressionDataset(
        df=train_df,
        feature_cols=feature_cols,
        clip_col=args.clip_col,
        x_std=x_std,
        y_std=y_std,
        fit_x=True,
        fit_y=True,
    )

    val_ds = SigmaRDRegressionDataset(
        df=val_df,
        feature_cols=feature_cols,
        clip_col=args.clip_col,
        x_std=x_std,
        y_std=y_std,
        fit_x=False,
        fit_y=False,
    )

    full_ds = SigmaRDRegressionDataset(
        df=long_df,
        feature_cols=feature_cols,
        clip_col=args.clip_col,
        x_std=x_std,
        y_std=y_std,
        fit_x=False,
        fit_y=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SmallMLP(
        in_dim=len(feature_cols) + 1,  # + sigma
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_score = -1e18
    best_epoch = -1
    best_ckpt_path = out_dir / f"best_qp{args.qp}.pth"

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        train_pred = predict_dataset(model, train_ds, device)
        train_metrics = summarize_predictions(train_pred)

        if len(val_ds) > 0:
            val_pred = predict_dataset(model, val_ds, device)
            val_metrics = summarize_predictions(val_pred)
            val_score = val_metrics["r2_r"] + val_metrics["r2_d"]
        else:
            val_metrics = {
                "mae_r": np.nan, "rmse_r": np.nan, "r2_r": np.nan,
                "mae_d": np.nan, "rmse_d": np.nan, "r2_d": np.nan,
            }
            val_score = train_metrics["r2_r"] + train_metrics["r2_d"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae_r": train_metrics["mae_r"],
            "train_rmse_r": train_metrics["rmse_r"],
            "train_r2_r": train_metrics["r2_r"],
            "train_mae_d": train_metrics["mae_d"],
            "train_rmse_d": train_metrics["rmse_d"],
            "train_r2_d": train_metrics["r2_d"],
            "val_mae_r": val_metrics["mae_r"],
            "val_rmse_r": val_metrics["rmse_r"],
            "val_r2_r": val_metrics["r2_r"],
            "val_mae_d": val_metrics["mae_d"],
            "val_rmse_d": val_metrics["rmse_d"],
            "val_r2_d": val_metrics["r2_d"],
        }
        log_rows.append(log_row)

        print(
            f"[{epoch:04d}/{args.epochs}] "
            f"loss={train_loss:.6f} | "
            f"train r2(R,D)=({train_metrics['r2_r']:.4f},{train_metrics['r2_d']:.4f}) | "
            f"val r2(R,D)=({val_metrics['r2_r']:.4f},{val_metrics['r2_d']:.4f})"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "x_std": train_ds.x_std.state_dict(),
                "y_std": train_ds.y_std.state_dict(),
                "feature_cols": feature_cols,
                "qp": args.qp,
                "sigma_tags": sigma_tags,
                "clip_col": args.clip_col,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "seed": args.seed,
            }
            torch.save(ckpt, best_ckpt_path)

    # save train log
    log_df = pd.DataFrame(log_rows)
    log_csv = out_dir / f"train_log_qp{args.qp}.csv"
    log_df.to_csv(log_csv, index=False)

    # load best
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # final predictions
    train_pred = predict_dataset(model, train_ds, device)
    val_pred = predict_dataset(model, val_ds, device) if len(val_ds) > 0 else pd.DataFrame()
    full_pred = predict_dataset(model, full_ds, device)

    train_metrics = summarize_predictions(train_pred)
    val_metrics = summarize_predictions(val_pred) if len(val_pred) > 0 else {
        "mae_r": np.nan, "rmse_r": np.nan, "r2_r": np.nan,
        "mae_d": np.nan, "rmse_d": np.nan, "r2_d": np.nan,
    }
    full_metrics = summarize_predictions(full_pred)

    train_pred.to_csv(out_dir / f"train_pred_qp{args.qp}.csv", index=False)
    if len(val_pred) > 0:
        val_pred.to_csv(out_dir / f"val_pred_qp{args.qp}.csv", index=False)
    full_pred.to_csv(out_dir / f"full_pred_qp{args.qp}.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_ckpt": str(best_ckpt_path),
        "qp": args.qp,
        "sigmas": sigma_tags,
        "n_train_rows": int(len(train_ds)),
        "n_val_rows": int(len(val_ds)),
        "n_full_rows": int(len(full_ds)),
        "n_feature_dims": int(len(feature_cols) + 1),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "full_metrics": full_metrics,
    }

    with open(out_dir / f"summary_qp{args.qp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"[DONE] best epoch: {best_epoch}")
    print(f"[DONE] checkpoint : {best_ckpt_path}")
    print(f"[DONE] log csv    : {log_csv}")
    print(f"[DONE] train pred : {out_dir / f'train_pred_qp{args.qp}.csv'}")
    if len(val_pred) > 0:
        print(f"[DONE] val pred   : {out_dir / f'val_pred_qp{args.qp}.csv'}")
    print(f"[DONE] full pred  : {out_dir / f'full_pred_qp{args.qp}.csv'}")
    print()
    print("[VAL]")
    print(
        f"R : MAE={val_metrics['mae_r']:.6f}, RMSE={val_metrics['rmse_r']:.6f}, R2={val_metrics['r2_r']:.6f}"
    )
    print(
        f"D : MAE={val_metrics['mae_d']:.6f}, RMSE={val_metrics['rmse_d']:.6f}, R2={val_metrics['r2_d']:.6f}"
    )


if __name__ == "__main__":
    main()
