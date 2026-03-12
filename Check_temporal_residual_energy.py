import os
import csv
import math
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Gaussian blur
# ------------------------------------------------------------
def gaussian_kernel_2d_fixed_k(sigma: float, k: int = 5, device="cpu", dtype=torch.float32):
    assert k % 2 == 1
    if sigma <= 0:
        ker = torch.zeros((1, 1, k, k), device=device, dtype=dtype)
        ker[..., k // 2, k // 2] = 1.0
        return ker
    ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker = ker / ker.sum()
    return ker.view(1, 1, k, k)


@torch.no_grad()
def blur_tchw_fixed5(x_tchw: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x_tchw: [T, C, H, W] in [0,1]
    Apply same 5x5 Gaussian per channel (grouped conv).
    """
    if sigma <= 0:
        return x_tchw
    T, C, H, W = x_tchw.shape
    ker = gaussian_kernel_2d_fixed_k(
        sigma, k=5, device=x_tchw.device, dtype=x_tchw.dtype
    )  # [1,1,5,5]
    w = ker.repeat(C, 1, 1, 1)  # [C,1,5,5]
    pad = 2
    xb = F.pad(x_tchw, (pad, pad, pad, pad), mode="reflect")
    yb = F.conv2d(xb, w, bias=None, stride=1, padding=0, groups=C)
    return yb


# ------------------------------------------------------------
# YUV420p10le reader: Y only
# ------------------------------------------------------------
def read_yuv420p10le_luma(path: str, width: int, height: int, num_frames: int) -> np.ndarray:
    """
    Read only luma from a yuv420p10le file.

    Returns:
        Y: np.ndarray of shape [T, H, W], dtype=np.uint16
    """
    path = str(path)
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height

    # 10-bit stored in 16-bit little-endian samples
    bytes_per_sample = 2
    frame_bytes = (y_size + uv_size + uv_size) * bytes_per_sample

    file_size = os.path.getsize(path)
    expected_size = frame_bytes * num_frames
    if file_size < expected_size:
        raise ValueError(
            f"File too small: {path}\n"
            f"Expected at least {expected_size} bytes for {num_frames} frames, got {file_size}"
        )

    Y = np.empty((num_frames, height, width), dtype=np.uint16)

    with open(path, "rb") as f:
        for t in range(num_frames):
            y = np.fromfile(f, dtype="<u2", count=y_size)
            if y.size != y_size:
                raise ValueError(f"Failed to read Y plane from {path}, frame {t}")
            Y[t] = y.reshape(height, width)

            # Skip U and V
            _ = np.fromfile(f, dtype="<u2", count=uv_size)
            _ = np.fromfile(f, dtype="<u2", count=uv_size)

    return Y


# ------------------------------------------------------------
# Motion compensation stub
# ------------------------------------------------------------
@torch.no_grad()
def motion_compensate_stub(target_1hw: torch.Tensor, reference_1hw: torch.Tensor) -> torch.Tensor:
    """
    target_1hw:    [1, H, W]
    reference_1hw: [1, H, W]

    TODO:
        Implement motion estimation + motion compensation here.
        Current placeholder returns identity reference,
        i.e., no motion compensation is applied.

    Return:
        compensated reference with shape [1, H, W]
    """
    # --------------------------------------------------------
    # Replace this with your motion estimation / compensation:
    #
    # 1) estimate motion from target <- reference
    # 2) warp / compensate reference toward target
    # 3) return compensated reference
    # --------------------------------------------------------
    return reference_1hw


# ------------------------------------------------------------
# Residual energy
# ------------------------------------------------------------
@torch.no_grad()
def residual_energy(target_1hw: torch.Tensor, pred_1hw: torch.Tensor) -> float:
    """
    Mean squared residual energy over all pixels.
    target_1hw, pred_1hw: [1, H, W], float32
    """
    res = target_1hw - pred_1hw
    return float((res * res).mean().item())


# ------------------------------------------------------------
# Clip analysis
# ------------------------------------------------------------
@torch.no_grad()
def analyze_clip(
    y_path: str,
    width: int,
    height: int,
    num_frames: int,
    sigma: float,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Analyze one clip:
      - original residual energy between t and t+1 after MC stub
      - blurred residual energy between blurred t and blurred t+1 after MC stub
      - compute reduction
    """
    Y_u16 = read_yuv420p10le_luma(y_path, width, height, num_frames)  # [T,H,W], uint16

    # Convert to [0,1] assuming 10-bit nominal range [0,1023]
    Y = torch.from_numpy(Y_u16.astype(np.float32) / 1023.0).to(device)  # [T,H,W]
    Y = Y.unsqueeze(1)  # [T,1,H,W]

    Y_blur = blur_tchw_fixed5(Y, sigma=sigma)  # [T,1,H,W]

    orig_energies: List[float] = []
    blur_energies: List[float] = []

    for t in range(num_frames - 1):
        target = Y[t]       # [1,H,W]
        reference = Y[t + 1]

        pred = motion_compensate_stub(target, reference)
        e_orig = residual_energy(target, pred)
        orig_energies.append(e_orig)

        target_b = Y_blur[t]
        reference_b = Y_blur[t + 1]

        pred_b = motion_compensate_stub(target_b, reference_b)
        e_blur = residual_energy(target_b, pred_b)
        blur_energies.append(e_blur)

    orig_mean = float(np.mean(orig_energies)) if orig_energies else 0.0
    blur_mean = float(np.mean(blur_energies)) if blur_energies else 0.0

    abs_reduction = orig_mean - blur_mean
    rel_reduction = abs_reduction / orig_mean if orig_mean > 0 else 0.0

    return {
        "clip_name": Path(y_path).stem,
        "sigma": float(sigma),
        "num_pairs": int(num_frames - 1),
        "orig_energy_mean": orig_mean,
        "blur_energy_mean": blur_mean,
        "abs_reduction": abs_reduction,
        "rel_reduction": rel_reduction,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def find_yuv_files(yuv_dir: str) -> List[Path]:
    exts = {".yuv"}
    files = [p for p in sorted(Path(yuv_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def save_results_csv(rows: List[Dict[str, float]], output_csv: str):
    if not rows:
        print("[WARN] No rows to save.")
        return

    fieldnames = [
        "clip_name",
        "sigma",
        "num_pairs",
        "orig_energy_mean",
        "blur_energy_mean",
        "abs_reduction",
        "rel_reduction",
    ]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yuv_dir", type=str, required=True, help="Directory containing yuv files")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    yuv_files = find_yuv_files(args.yuv_dir)
    if not yuv_files:
        raise FileNotFoundError(f"No .yuv files found in: {args.yuv_dir}")

    results = []
    for i, yuv_path in enumerate(yuv_files, 1):
        print(f"[{i}/{len(yuv_files)}] Processing: {yuv_path.name}")
        row = analyze_clip(
            y_path=str(yuv_path),
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            sigma=args.sigma,
            device=args.device,
        )
        results.append(row)

    save_results_csv(results, args.output_csv)
    print(f"[INFO] Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()





















import numpy as np


def hadamard8x8(block):
    H = np.array([
        [1,1,1,1,1,1,1,1],
        [1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1],
        [1,1,1,1,-1,-1,-1,-1],
        [1,-1,1,-1,-1,1,-1,1],
        [1,1,-1,-1,-1,-1,1,1],
        [1,-1,-1,1,-1,1,1,-1],
    ], dtype=np.float32)

    return H @ block @ H.T


def satd8x8(block):
    t = hadamard8x8(block)
    return np.sum(np.abs(t)) / 64.0


def frame_satd(residual, block=8):
    h, w = residual.shape
    s = 0.0

    for y in range(0, h, block):
        for x in range(0, w, block):
            b = residual[y:y+block, x:x+block]
            if b.shape == (block, block):
                s += satd8x8(b)

    return s


def simple_motion_estimation(cur, ref, search=4):
    h, w = cur.shape
    best_dx = 0
    best_dy = 0
    best_cost = 1e18

    for dy in range(-search, search+1):
        for dx in range(-search, search+1):

            shifted = np.roll(ref, (dy, dx), axis=(0,1))
            cost = np.mean(np.abs(cur - shifted))

            if cost < best_cost:
                best_cost = cost
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy


def motion_compensate(ref, dx, dy):
    return np.roll(ref, (dy, dx), axis=(0,1))


def find_ra_refs(idx, processed):
    left = None
    right = None

    for p in processed:
        if p < idx:
            if left is None or p > left:
                left = p
        elif p > idx:
            if right is None or p < right:
                right = p

    return left, right


def ra_temporal_satd(frames, order):
    """
    frames: list of grayscale frames (numpy HxW)
    order : RA coding order list
    """

    processed = []
    energy = {}

    for t in order:

        if len(processed) < 2:
            processed.append(t)
            energy[t] = 0
            continue

        left, right = find_ra_refs(t, processed)

        if left is None or right is None:
            processed.append(t)
            energy[t] = 0
            continue

        cur = frames[t]

        refL = frames[left]
        refR = frames[right]

        dxL, dyL = simple_motion_estimation(cur, refL)
        dxR, dyR = simple_motion_estimation(cur, refR)

        mcL = motion_compensate(refL, dxL, dyL)
        mcR = motion_compensate(refR, dxR, dyR)

        pred = (mcL.astype(np.float32) + mcR.astype(np.float32)) * 0.5

        residual = cur.astype(np.float32) - pred

        satd = frame_satd(residual)

        energy[t] = satd

        processed.append(t)

    return energy
