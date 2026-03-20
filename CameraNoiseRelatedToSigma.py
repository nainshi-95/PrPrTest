#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Fixed input format
# ============================================================
WIDTH = 256
HEIGHT = 256
FRAMES = 33
BIT_DEPTH = 10


# ============================================================
# YUV420p10le IO
# ============================================================
def read_yuv420p10le(
    path: Path,
    width: int = WIDTH,
    height: int = HEIGHT,
    frames: int = FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Y: [T,H,W] uint16
      U: [T,H/2,W/2] uint16
      V: [T,H/2,W/2] uint16
    """
    w2, h2 = width // 2, height // 2
    y_n = width * height
    uv_n = w2 * h2

    Y = np.empty((frames, height, width), dtype=np.uint16)
    U = np.empty((frames, h2, w2), dtype=np.uint16)
    V = np.empty((frames, h2, w2), dtype=np.uint16)

    with open(path, "rb") as f:
        for t in range(frames):
            yb = f.read(y_n * 2)
            ub = f.read(uv_n * 2)
            vb = f.read(uv_n * 2)

            if len(yb) != y_n * 2 or len(ub) != uv_n * 2 or len(vb) != uv_n * 2:
                raise IOError(f"EOF while reading {path} at frame {t}")

            Y[t] = np.frombuffer(yb, dtype=np.dtype("<u2")).reshape(height, width)
            U[t] = np.frombuffer(ub, dtype=np.dtype("<u2")).reshape(h2, w2)
            V[t] = np.frombuffer(vb, dtype=np.dtype("<u2")).reshape(h2, w2)

    return Y, U, V


def to_float01_u10(x_u16: np.ndarray) -> np.ndarray:
    return np.clip(x_u16.astype(np.float32) / 1023.0, 0.0, 1.0)


# ============================================================
# Model
# ============================================================
def build_model():
    """
    Replace this with your actual preprocessor loader.

    Expected:
        y_out, u_out, v_out = model(y, u, v)

    Inputs:
        y: [B,T,1,H,W]
        u: [B,T,1,H/2,W/2]
        v: [B,T,1,H/2,W/2]

    Outputs:
        same shapes
    """
    raise NotImplementedError("Replace build_model() with your actual model loader.")


# ============================================================
# Spatial pad to multiple of 16
# ============================================================
def pad_btchw_to_multiple_of_16(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    x: [B,T,C,H,W]
    """
    B, T, C, H, W = x.shape
    pad_h = (16 - (H % 16)) % 16
    pad_w = (16 - (W % 16)) % 16

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    x2 = x.reshape(B * T, C, H, W)
    x2 = F.pad(x2, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = x2.shape[-2:]
    x2 = x2.reshape(B, T, C, Hp, Wp)
    return x2, (pad_h, pad_w)


def crop_btchw(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    H = x.shape[-2]
    W = x.shape[-1]
    return x[..., :H - pad_h, :W - pad_w]


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def run_preprocessor_fullclip(
    model: torch.nn.Module,
    Y: np.ndarray,   # [T,H,W], float01
    U: np.ndarray,   # [T,H/2,W/2], float01
    V: np.ndarray,   # [T,H/2,W/2], float01
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full clip inference.

    Inputs:
      Y: [T,H,W]
      U: [T,H/2,W/2]
      V: [T,H/2,W/2]

    Returns:
      Yo: [T,H,W]
      Uo: [T,H/2,W/2]
      Vo: [T,H/2,W/2]
    """
    y = torch.from_numpy(Y).to(device).unsqueeze(0).unsqueeze(2)  # [1,T,1,H,W]
    u = torch.from_numpy(U).to(device).unsqueeze(0).unsqueeze(2)  # [1,T,1,H/2,W/2]
    v = torch.from_numpy(V).to(device).unsqueeze(0).unsqueeze(2)  # [1,T,1,H/2,W/2]

    y, (py, px) = pad_btchw_to_multiple_of_16(y)
    u, (puy, pux) = pad_btchw_to_multiple_of_16(u)
    v, (pvy, pvx) = pad_btchw_to_multiple_of_16(v)

    y_out, u_out, v_out = model(y, u, v)

    y_out = crop_btchw(y_out, py, px)
    u_out = crop_btchw(u_out, puy, pux)
    v_out = crop_btchw(v_out, pvy, pvx)

    # expected [1,T,1,H,W] -> [T,H,W]
    y_out = y_out.squeeze(0).squeeze(1).detach().float().cpu().numpy()
    u_out = u_out.squeeze(0).squeeze(1).detach().float().cpu().numpy()
    v_out = v_out.squeeze(0).squeeze(1).detach().float().cpu().numpy()

    y_out = np.clip(y_out, 0.0, 1.0)
    u_out = np.clip(u_out, 0.0, 1.0)
    v_out = np.clip(v_out, 0.0, 1.0)

    return y_out, u_out, v_out


# ============================================================
# File utils
# ============================================================
def find_yuv_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.yuv"))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_root", type=str, required=True, help="Root folder containing 256x256 33f 10bit yuv files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    input_root = Path(args.input_root)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    yuv_files = find_yuv_files(input_root)
    if not yuv_files:
        raise FileNotFoundError(f"No .yuv files found under: {input_root}")

    model = build_model()
    model = model.to(device)
    model.eval()

    print("filename,y_mean,u_mean,v_mean")

    for yuv_path in yuv_files:
        try:
            Y_u16, U_u16, V_u16 = read_yuv420p10le(
                yuv_path,
                width=WIDTH,
                height=HEIGHT,
                frames=FRAMES,
            )

            Y = to_float01_u10(Y_u16)
            U = to_float01_u10(U_u16)
            V = to_float01_u10(V_u16)

            Yo, Uo, Vo = run_preprocessor_fullclip(
                model=model,
                Y=Y,
                U=U,
                V=V,
                device=device,
            )

            y_mean = float(np.mean(Yo))
            u_mean = float(np.mean(Uo))
            v_mean = float(np.mean(Vo))

            print(f"{yuv_path.name},{y_mean:.8f},{u_mean:.8f},{v_mean:.8f}")

        except Exception as e:
            print(f"{yuv_path.name},ERROR,{e}")


if __name__ == "__main__":
    main()
