#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_hidden_dims(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty --hidden_dims")
    return vals


def find_clip_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in ["clip_name", "clip_id", "name"]:
        if cand in lower:
            return lower[cand]
    # 첫 번째 column fallback
    return df.columns[0]


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def acc_tol_np(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    if n_test <= 0:
        raise ValueError("Not enough samples for test split")

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ============================================================
# Dataset
# ============================================================
class SigmaDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================
# Model
# ============================================================
class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_n = 0
    all_y = []
    all_p = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y, axis=0).reshape(-1)
    y_pred = np.concatenate(all_p, axis=0).reshape(-1)

    return {
        "loss": total_loss / max(total_n, 1),
        "mae": mae_np(y_true, y_pred),
        "rmse": rmse_np(y_true, y_pred),
        "r2": r2_np(y_true, y_pred),
        "acc_tol_0.025": acc_tol_np(y_true, y_pred, 0.025),
        "acc_tol_0.05": acc_tol_np(y_true, y_pred, 0.05),
        "acc_tol_0.10": acc_tol_np(y_true, y_pred, 0.10),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_csv", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--qp", type=int, required=True, help="target column: best_sigma_qpXX")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--hidden_dims", type=str, default="16,8")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    target_col = f"best_sigma_qp{args.qp}"

    out_dir = Path(args.out_dir) / f"qp{args.qp}"
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)

    # ------------------------------------------------------------
    # Load and merge
    # ------------------------------------------------------------
    df_x = pd.read_csv(args.dataset_csv)
    df_y = pd.read_csv(args.gt_csv)

    clip_col_x = find_clip_col(df_x)
    clip_col_y = find_clip_col(df_y)

    if target_col not in df_y.columns:
        raise KeyError(f"Missing target column: {target_col}")

    df = pd.merge(
        df_x,
        df_y[[clip_col_y, target_col]].copy(),
        left_on=clip_col_x,
        right_on=clip_col_y,
        how="inner",
    )

    if len(df) == 0:
        raise RuntimeError("Merged dataframe is empty")

    # numeric feature selection
    exclude_cols = {clip_col_x, clip_col_y, target_col}
    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        try:
            pd.to_numeric(df[c], errors="raise")
            feature_cols.append(c)
        except Exception:
            pass

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found")

    x = df[feature_cols].astype(np.float32).to_numpy()
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(np.float32)
    clip_names = df[clip_col_x].astype(str).to_numpy()

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    clip_names = clip_names[valid]

    if len(y) < 10:
        raise RuntimeError(f"Too few valid samples: {len(y)}")

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------
    train_idx, val_idx, test_idx = split_indices(
        n=len(y),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    clip_train = clip_names[train_idx]
    clip_val = clip_names[val_idx]
    clip_test = clip_names[test_idx]

    # normalize using train only
    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_val = standardize_apply(x_val, mean, std)
    x_test = standardize_apply(x_test, mean, std)

    ds_train = SigmaDataset(x_train, y_train)
    ds_val = SigmaDataset(x_val, y_val)
    ds_test = SigmaDataset(x_test, y_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    model = SmallMLP(
        in_dim=x_train.shape[1],
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    best_val_loss = math.inf
    best_epoch = -1
    patience_count = 0
    history = []

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dl_train, optimizer, device)
        val_metrics = eval_model(model, dl_val, device)

        hist = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_acc_tol_0.025": val_metrics["acc_tol_0.025"],
            "val_acc_tol_0.05": val_metrics["acc_tol_0.05"],
            "val_acc_tol_0.10": val_metrics["acc_tol_0.10"],
        }
        history.append(hist)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "target_col": target_col,
            "hidden_dims": parse_hidden_dims(args.hidden_dims),
            "dropout": args.dropout,
        }, last_ckpt)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_count = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "feature_cols": feature_cols,
                "mean": mean,
                "std": std,
                "target_col": target_col,
                "hidden_dims": parse_hidden_dims(args.hidden_dims),
                "dropout": args.dropout,
            }, best_ckpt)
        else:
            patience_count += 1

        print(
            f"[Epoch {epoch:03d}] "
            f"train={train_loss:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"mae={val_metrics['mae']:.6f} "
            f"rmse={val_metrics['rmse']:.6f} "
            f"r2={val_metrics['r2']:.6f}"
        )

        if patience_count >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}, best epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    # ------------------------------------------------------------
    # Best checkpoint eval
    # ------------------------------------------------------------
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_metrics = eval_model(model, dl_train, device)
    val_metrics = eval_model(model, dl_val, device)
    test_metrics = eval_model(model, dl_test, device)

    metrics = {
        "qp": args.qp,
        "target_col": target_col,
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),

        "train_loss": train_metrics["loss"],
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],

        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],

        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_acc_tol_0.025": test_metrics["acc_tol_0.025"],
        "test_acc_tol_0.05": test_metrics["acc_tol_0.05"],
        "test_acc_tol_0.10": test_metrics["acc_tol_0.10"],
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # test prediction table
    pred_df = pd.DataFrame({
        "clip_name": clip_test,
        "gt_sigma": test_metrics["y_true"],
        "pred_sigma": test_metrics["y_pred"],
        "abs_err": np.abs(test_metrics["y_true"] - test_metrics["y_pred"]),
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # split info
    split_df = pd.DataFrame({
        "clip_name": np.concatenate([clip_train, clip_val, clip_test]),
        "split": (
            ["train"] * len(clip_train) +
            ["val"] * len(clip_val) +
            ["test"] * len(clip_test)
        )
    })
    split_df.to_csv(out_dir / "split.csv", index=False)

    print("\n[Best checkpoint evaluation]")
    print(f"QP                 : {args.qp}")
    print(f"Best epoch         : {best_epoch}")
    print(f"Train MAE / RMSE   : {train_metrics['mae']:.6f} / {train_metrics['rmse']:.6f}")
    print(f"Val   MAE / RMSE   : {val_metrics['mae']:.6f} / {val_metrics['rmse']:.6f}")
    print(f"Test  MAE / RMSE   : {test_metrics['mae']:.6f} / {test_metrics['rmse']:.6f}")
    print(f"Test  R2           : {test_metrics['r2']:.6f}")
    print(f"Test acc |e|<=0.025: {test_metrics['acc_tol_0.025']:.4f}")
    print(f"Test acc |e|<=0.05 : {test_metrics['acc_tol_0.05']:.4f}")
    print(f"Test acc |e|<=0.10 : {test_metrics['acc_tol_0.10']:.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
