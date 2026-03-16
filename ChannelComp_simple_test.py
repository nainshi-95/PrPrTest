import os
import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================
def psnr_from_float01(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    a, b: float arrays in [0, 1]
    """
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse < eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_dtype_and_max(bit_depth: int):
    if bit_depth == 8:
        return np.uint8, 255
    elif bit_depth == 10:
        return np.uint16, 1023
    else:
        raise ValueError(f"Unsupported bit_depth: {bit_depth}")


# ============================================================
# YUV Reader
# ============================================================
class YUV420Reader:
    def __init__(self, path: str, width: int, height: int, bit_depth: int):
        self.path = path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth

        self.uv_width = width // 2
        self.uv_height = height // 2

        self.n_y = self.width * self.height
        self.n_uv = self.uv_width * self.uv_height

        if bit_depth == 8:
            self.dtype = np.uint8
            self.bytes_per_sample = 1
            self.max_val = 255
        elif bit_depth == 10:
            # yuv420p10le is typically stored in 16-bit little-endian containers
            self.dtype = np.uint16
            self.bytes_per_sample = 2
            self.max_val = 1023
        else:
            raise ValueError("bit_depth must be 8 or 10")

        self.frame_bytes = (self.n_y + 2 * self.n_uv) * self.bytes_per_sample
        self.f = open(self.path, "rb")

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def read_frame(self):
        raw = self.f.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return None

        arr = np.frombuffer(raw, dtype=self.dtype)

        y = arr[:self.n_y].reshape(self.height, self.width)
        u = arr[self.n_y:self.n_y + self.n_uv].reshape(self.uv_height, self.uv_width)
        v = arr[self.n_y + self.n_uv:self.n_y + 2 * self.n_uv].reshape(self.uv_height, self.uv_width)

        return y, u, v

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ============================================================
# Writers
# ============================================================
class Raw1Ch10BitWriter:
    """
    Writes B,1,H,W single-channel output to raw 10-bit (stored as uint16 LE).
    """
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")

    def write_frame(self, x01: np.ndarray):
        """
        x01: H,W float in [0,1]
        """
        x10 = np.clip(np.round(x01 * 1023.0), 0, 1023).astype(np.uint16)
        self.f.write(x10.tobytes())

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class YUV420Writer10Bit:
    """
    Writes YUV420 10-bit raw (uint16 LE).
    """
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")

    def write_frame(self, y01: np.ndarray, u01: np.ndarray, v01: np.ndarray):
        y10 = np.clip(np.round(y01 * 1023.0), 0, 1023).astype(np.uint16)
        u10 = np.clip(np.round(u01 * 1023.0), 0, 1023).astype(np.uint16)
        v10 = np.clip(np.round(v01 * 1023.0), 0, 1023).astype(np.uint16)

        self.f.write(y10.tobytes())
        self.f.write(u10.tobytes())
        self.f.write(v10.tobytes())

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ============================================================
# Visualization
# ============================================================
def save_visualization(
    out_png: str,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    y_rec: np.ndarray,
    u_rec: np.ndarray,
    v_rec: np.ndarray,
    frame_idx: int,
    psnr_y: float,
    psnr_u: float,
    psnr_v: float,
):
    """
    All inputs are float [0,1].
    """
    fig = plt.figure(figsize=(12, 8))

    ax = plt.subplot(2, 4, 1)
    ax.imshow(y, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Input Y (f={frame_idx})")
    ax.axis("off")

    ax = plt.subplot(2, 4, 2)
    ax.imshow(u, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Input U")
    ax.axis("off")

    ax = plt.subplot(2, 4, 3)
    ax.imshow(v, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Input V")
    ax.axis("off")

    ax = plt.subplot(2, 4, 4)
    ax.imshow(t, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Transform (1ch)")
    ax.axis("off")

    ax = plt.subplot(2, 4, 5)
    ax.imshow(y_rec, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Rec Y\nPSNR={psnr_y:.4f} dB")
    ax.axis("off")

    ax = plt.subplot(2, 4, 6)
    ax.imshow(u_rec, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Rec U\nPSNR={psnr_u:.4f} dB")
    ax.axis("off")

    ax = plt.subplot(2, 4, 7)
    ax.imshow(v_rec, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Rec V\nPSNR={psnr_v:.4f} dB")
    ax.axis("off")

    diff_y = np.abs(y - y_rec)
    ax = plt.subplot(2, 4, 8)
    ax.imshow(diff_y, cmap="hot", vmin=0, vmax=max(1e-6, diff_y.max()))
    ax.set_title("Abs Error Y")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================
# Example dummy model
# Replace this with your real model
# ============================================================
class DummyModel(torch.nn.Module):
    def transform(self, y, u, v):
        # Example only: just return Y
        return y

    def itransform(self, x):
        # Example only: reconstruct YUV from x
        b, c, h, w = x.shape
        u = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="area")
        v = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="area")
        return x, u, v


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input YUV path")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--bit_depth", type=int, choices=[8, 10], required=True)
    parser.add_argument("--format", type=str, choices=["yuv420p", "yuv420p10le"], required=True)

    parser.add_argument("--out_transform", type=str, default="transform_1ch_10bit.raw")
    parser.add_argument("--out_inverse", type=str, default="inverse_rec_420p10le.yuv")
    parser.add_argument("--vis_dir", type=str, default="vis")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Basic format-depth consistency check
    if args.format == "yuv420p" and args.bit_depth != 8:
        raise ValueError("format=yuv420p must use bit_depth=8")
    if args.format == "yuv420p10le" and args.bit_depth != 10:
        raise ValueError("format=yuv420p10le must use bit_depth=10")

    ensure_dir(args.vis_dir)

    device = torch.device(args.device)

    # --------------------------------------------------------
    # Replace this with your actual model loading code
    # --------------------------------------------------------
    model = DummyModel().to(device).eval()

    psnr_y_list = []
    psnr_u_list = []
    psnr_v_list = []

    with YUV420Reader(args.input, args.width, args.height, args.bit_depth) as reader, \
         Raw1Ch10BitWriter(args.out_transform) as tf_writer, \
         YUV420Writer10Bit(args.out_inverse) as inv_writer:

        for frame_idx in range(args.num_frames):
            frame = reader.read_frame()
            if frame is None:
                print(f"[INFO] Reached EOF at frame {frame_idx}")
                break

            y_np_i, u_np_i, v_np_i = frame
            max_in = 255.0 if args.bit_depth == 8 else 1023.0

            # float [0,1]
            y_np = y_np_i.astype(np.float32) / max_in
            u_np = u_np_i.astype(np.float32) / max_in
            v_np = v_np_i.astype(np.float32) / max_in

            # to tensor: B,1,H,W
            y = torch.from_numpy(y_np)[None, None].to(device)
            u = torch.from_numpy(u_np)[None, None].to(device)
            v = torch.from_numpy(v_np)[None, None].to(device)

            with torch.no_grad():
                t = model.transform(y, u, v)
                if not isinstance(t, torch.Tensor):
                    raise TypeError("model.transform(...) must return a torch.Tensor")
                if t.ndim != 4 or t.shape[0] != 1 or t.shape[1] != 1 or t.shape[2] != args.height or t.shape[3] != args.width:
                    raise ValueError(
                        f"transform output must be [1,1,H,W], got {tuple(t.shape)}"
                    )

                t = torch.clamp(t, 0.0, 1.0)

                y_rec, u_rec, v_rec = model.itransform(t)

                for name, ten, hh, ww in [
                    ("y_rec", y_rec, args.height, args.width),
                    ("u_rec", u_rec, args.height // 2, args.width // 2),
                    ("v_rec", v_rec, args.height // 2, args.width // 2),
                ]:
                    if not isinstance(ten, torch.Tensor):
                        raise TypeError(f"{name} must be a torch.Tensor")
                    if ten.ndim != 4 or ten.shape[0] != 1 or ten.shape[1] != 1 or ten.shape[2] != hh or ten.shape[3] != ww:
                        raise ValueError(
                            f"{name} must be [1,1,{hh},{ww}], got {tuple(ten.shape)}"
                        )

                y_rec = torch.clamp(y_rec, 0.0, 1.0)
                u_rec = torch.clamp(u_rec, 0.0, 1.0)
                v_rec = torch.clamp(v_rec, 0.0, 1.0)

            # to numpy
            t_np = t[0, 0].detach().cpu().numpy()
            y_rec_np = y_rec[0, 0].detach().cpu().numpy()
            u_rec_np = u_rec[0, 0].detach().cpu().numpy()
            v_rec_np = v_rec[0, 0].detach().cpu().numpy()

            # save raw outputs
            tf_writer.write_frame(t_np)
            inv_writer.write_frame(y_rec_np, u_rec_np, v_rec_np)

            # PSNR
            psnr_y = psnr_from_float01(y_np, y_rec_np)
            psnr_u = psnr_from_float01(u_np, u_rec_np)
            psnr_v = psnr_from_float01(v_np, v_rec_np)

            psnr_y_list.append(psnr_y)
            psnr_u_list.append(psnr_u)
            psnr_v_list.append(psnr_v)

            # visualization
            vis_path = os.path.join(args.vis_dir, f"frame_{frame_idx:04d}.png")
            save_visualization(
                vis_path,
                y_np,
                u_np,
                v_np,
                t_np,
                y_rec_np,
                u_rec_np,
                v_rec_np,
                frame_idx,
                psnr_y,
                psnr_u,
                psnr_v,
            )

            print(
                f"[Frame {frame_idx:04d}] "
                f"PSNR-Y: {psnr_y:.4f} dB, "
                f"PSNR-U: {psnr_u:.4f} dB, "
                f"PSNR-V: {psnr_v:.4f} dB"
            )

    if len(psnr_y_list) > 0:
        print("\n=== Average PSNR ===")
        print(f"Y: {sum(psnr_y_list)/len(psnr_y_list):.4f} dB")
        print(f"U: {sum(psnr_u_list)/len(psnr_u_list):.4f} dB")
        print(f"V: {sum(psnr_v_list)/len(psnr_v_list):.4f} dB")
    else:
        print("No frame processed.")


if __name__ == "__main__":
    main()
