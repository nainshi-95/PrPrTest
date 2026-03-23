#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# YUV loader (420p10le)
# ============================================================
def read_yuv420p10le(path: Path, w=256, h=256, frames=33):
    w2, h2 = w // 2, h // 2

    y_n = w * h
    uv_n = w2 * h2
    per_frame = y_n + uv_n * 2
    total = per_frame * frames

    data = np.fromfile(path, dtype=np.uint16, count=total)
    if data.size != total:
        raise RuntimeError(f"{path} size mismatch")

    data = data.astype(np.float32) / 1023.0

    Y = np.empty((frames, h, w), dtype=np.float32)

    idx = 0
    for t in range(frames):
        Y[t] = data[idx:idx + y_n].reshape(h, w)
        idx += y_n + uv_n * 2  # skip UV

    return Y


# ============================================================
# Feature extraction
# ============================================================
def sobel_mag(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    return np.sqrt(gx**2 + gy**2)


def laplacian(img):
    lap = (
        -4 * img
        + np.roll(img, 1, 0)
        + np.roll(img, -1, 0)
        + np.roll(img, 1, 1)
        + np.roll(img, -1, 1)
    )
    return lap


def extract_features(Y):
    T, H, W = Y.shape

    # -----------------------------
    # spatial (frame-wise)
    # -----------------------------
    mean_vals = []
    var_vals = []
    grad_vals = []
    lap_vals = []

    for t in range(T):
        f = Y[t]

        mean_vals.append(np.mean(f))
        var_vals.append(np.var(f))

        g = sobel_mag(f)
        grad_vals.append(np.mean(g))

        l = laplacian(f)
        lap_vals.append(np.mean(np.abs(l)))

    # -----------------------------
    # temporal
    # -----------------------------
    diff_vals = []
    for t in range(1, T):
        diff = Y[t] - Y[t - 1]
        diff_vals.append(np.mean(diff**2))

    # -----------------------------
    # aggregation
    # -----------------------------
    feat = {}

    # spatial
    feat["mean"] = float(np.mean(mean_vals))
    feat["var"] = float(np.mean(var_vals))
    feat["grad"] = float(np.mean(grad_vals))
    feat["lap"] = float(np.mean(lap_vals))

    # temporal
    feat["ti"] = float(np.mean(diff_vals)) if diff_vals else 0.0
    feat["temporal_var"] = float(np.var(mean_vals))

    # distribution
    all_pixels = Y.reshape(-1)
    feat["p10"] = float(np.percentile(all_pixels, 10))
    feat["p90"] = float(np.percentile(all_pixels, 90))

    return feat


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yuv_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=33)

    args = parser.parse_args()

    root = Path(args.yuv_root)
    yuv_files = sorted(root.rglob("*.yuv"))

    if not yuv_files:
        raise RuntimeError("No YUV files found")

    rows = []

    for i, yuv_path in enumerate(yuv_files, 1):
        print(f"[{i}/{len(yuv_files)}] {yuv_path.name}")

        try:
            Y = read_yuv420p10le(
                yuv_path,
                w=args.width,
                h=args.height,
                frames=args.frames,
            )

            feat = extract_features(Y)

            row = {"clip_name": yuv_path.stem}
            row.update(feat)

            rows.append(row)

        except Exception as e:
            print(f"[WARN] failed: {yuv_path} ({e})")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print(f"[OK] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
