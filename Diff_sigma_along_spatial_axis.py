import os
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
    ker = gaussian_kernel_2d_fixed_k(sigma, k=5, device=x_tchw.device, dtype=x_tchw.dtype)  # [1,1,5,5]
    w = ker.repeat(C, 1, 1, 1)  # [C,1,5,5]
    pad = 2
    xb = F.pad(x_tchw, (pad, pad, pad, pad), mode="reflect")
    yb = F.conv2d(xb, w, bias=None, stride=1, padding=0, groups=C)
    return yb


def read_yuv420(path: str, width: int, height: int, num_frames: int, bit_depth: int):
    """
    Returns:
        Y: [T, H, W]
        U: [T, H/2, W/2]
        V: [T, H/2, W/2]
    """
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    if bit_depth == 8:
        dtype = np.uint8
        bytes_per_sample = 1
        max_val = 255
    elif bit_depth == 10:
        dtype = np.uint16
        bytes_per_sample = 2
        max_val = 1023
    else:
        raise ValueError("bit_depth must be 8 or 10.")

    w_uv = width // 2
    h_uv = height // 2

    y_samples = width * height
    uv_samples = w_uv * h_uv
    frame_samples = y_samples + uv_samples + uv_samples
    frame_bytes = frame_samples * bytes_per_sample

    file_size = os.path.getsize(path)
    max_frames_in_file = file_size // frame_bytes
    if num_frames > max_frames_in_file:
        raise ValueError(
            f"Requested num_frames={num_frames}, but file only contains about {max_frames_in_file} frames."
        )

    Y = np.empty((num_frames, height, width), dtype=dtype)
    U = np.empty((num_frames, h_uv, w_uv), dtype=dtype)
    V = np.empty((num_frames, h_uv, w_uv), dtype=dtype)

    with open(path, "rb") as f:
        for t in range(num_frames):
            y = np.fromfile(f, dtype=dtype, count=y_samples)
            u = np.fromfile(f, dtype=dtype, count=uv_samples)
            v = np.fromfile(f, dtype=dtype, count=uv_samples)

            if y.size != y_samples or u.size != uv_samples or v.size != uv_samples:
                raise RuntimeError(f"Unexpected EOF while reading frame {t}.")

            Y[t] = y.reshape(height, width)
            U[t] = u.reshape(h_uv, w_uv)
            V[t] = v.reshape(h_uv, w_uv)

    return Y, U, V, max_val


def write_yuv420(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray, bit_depth: int):
    if bit_depth == 8:
        out_dtype = np.uint8
    elif bit_depth == 10:
        out_dtype = np.uint16
    else:
        raise ValueError("bit_depth must be 8 or 10.")

    with open(path, "wb") as f:
        T = Y.shape[0]
        for t in range(T):
            Y[t].astype(out_dtype).tofile(f)
            U[t].astype(out_dtype).tofile(f)
            V[t].astype(out_dtype).tofile(f)


def reflect_pad_to_multiple(x_tchw: torch.Tensor, block_h: int, block_w: int):
    """
    x_tchw: [T, 1, H, W]
    pad right/bottom so H and W become multiples of block_h/block_w
    """
    T, C, H, W = x_tchw.shape
    pad_h = (block_h - (H % block_h)) % block_h
    pad_w = (block_w - (W % block_w)) % block_w

    if pad_h == 0 and pad_w == 0:
        return x_tchw, H, W

    x_pad = F.pad(x_tchw, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, H, W


@torch.no_grad()
def apply_blockwise_blur_same_over_time(
    x_tchw: torch.Tensor,
    sigma_map: np.ndarray,
    block_h: int,
    block_w: int,
):
    """
    x_tchw: [T, 1, H, W] in [0,1]
    sigma_map: [nby, nbx], same sigma across time for each spatial block
    """
    x_pad, orig_h, orig_w = reflect_pad_to_multiple(x_tchw, block_h, block_w)
    T, C, Hpad, Wpad = x_pad.shape

    nby = Hpad // block_h
    nbx = Wpad // block_w
    if sigma_map.shape != (nby, nbx):
        raise ValueError(
            f"sigma_map shape mismatch: expected {(nby, nbx)}, got {sigma_map.shape}"
        )

    out = torch.empty_like(x_pad)

    for by in range(nby):
        y0 = by * block_h
        y1 = y0 + block_h
        for bx in range(nbx):
            x0 = bx * block_w
            x1 = x0 + block_w

            sigma = float(sigma_map[by, bx])
            patch = x_pad[:, :, y0:y1, x0:x1]  # [T,1,block_h,block_w]
            out[:, :, y0:y1, x0:x1] = blur_tchw_fixed5(patch, sigma)

    return out[:, :, :orig_h, :orig_w]


def make_sigma_map(height: int, width: int, clip_size: int, sigma_min: float, sigma_max: float, seed: int):
    nby = math.ceil(height / clip_size)
    nbx = math.ceil(width / clip_size)
    rng = np.random.default_rng(seed)
    sigma_map = rng.uniform(sigma_min, sigma_max, size=(nby, nbx)).astype(np.float32)
    return sigma_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input YUV file")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--bit_depth", type=int, choices=[8, 10], required=True)
    parser.add_argument("--clip_size", type=int, required=True, help="Spatial block size on luma plane")
    parser.add_argument("--sigma_min", type=float, required=True)
    parser.add_argument("--sigma_max", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.width % 2 != 0 or args.height % 2 != 0:
        raise ValueError("For YUV420, width and height must be even.")
    if args.clip_size <= 0:
        raise ValueError("clip_size must be positive.")
    if args.clip_size % 2 != 0:
        raise ValueError("For YUV420 handling, clip_size must be even.")
    if args.sigma_min < 0 or args.sigma_max < 0:
        raise ValueError("sigma_min and sigma_max must be >= 0.")
    if args.sigma_min > args.sigma_max:
        raise ValueError("sigma_min must be <= sigma_max.")

    os.makedirs(args.output_dir, exist_ok=True)

    input_path = Path(args.input)
    stem = input_path.stem

    if args.bit_depth == 8:
        pixfmt = "yuv420p"
        ext_tag = "8b"
    else:
        pixfmt = "yuv420p10le"
        ext_tag = "10b"

    out_orig = Path(args.output_dir) / f"{stem}_{args.width}x{args.height}_{args.num_frames}f_{ext_tag}_orig.yuv"
    out_blur = Path(args.output_dir) / (
        f"{stem}_{args.width}x{args.height}_{args.num_frames}f_"
        f"{ext_tag}_clip{args.clip_size}_sigma{args.sigma_min:g}-{args.sigma_max:g}_blur.yuv"
    )

    print(f"[Info] Reading input: {args.input}")
    print(f"[Info] Format: {pixfmt}")
    Y_np, U_np, V_np, max_val = read_yuv420(
        path=args.input,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        bit_depth=args.bit_depth,
    )

    # Write original selected clip
    print(f"[Info] Writing original clip: {out_orig}")
    write_yuv420(str(out_orig), Y_np, U_np, V_np, args.bit_depth)

    # Normalize to [0,1], move to torch
    device = torch.device(args.device)
    Y = torch.from_numpy(Y_np.astype(np.float32) / max_val).unsqueeze(1).to(device)  # [T,1,H,W]
    U = torch.from_numpy(U_np.astype(np.float32) / max_val).unsqueeze(1).to(device)  # [T,1,H/2,W/2]
    V = torch.from_numpy(V_np.astype(np.float32) / max_val).unsqueeze(1).to(device)  # [T,1,H/2,W/2]

    # Sigma map on luma clip grid
    sigma_map_y = make_sigma_map(
        height=args.height,
        width=args.width,
        clip_size=args.clip_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        seed=args.seed,
    )

    # Same sigma structure for chroma, using corresponding half-resolution block size
    clip_uv = args.clip_size // 2
    sigma_map_uv = sigma_map_y.copy()

    print(f"[Info] Applying block-wise blur...")
    print(f"[Info] Luma sigma map shape: {sigma_map_y.shape}")
    print(f"[Info] Sigma range: [{sigma_map_y.min():.4f}, {sigma_map_y.max():.4f}]")

    Y_blur = apply_blockwise_blur_same_over_time(
        x_tchw=Y,
        sigma_map=sigma_map_y,
        block_h=args.clip_size,
        block_w=args.clip_size,
    )
    U_blur = apply_blockwise_blur_same_over_time(
        x_tchw=U,
        sigma_map=sigma_map_uv,
        block_h=clip_uv,
        block_w=clip_uv,
    )
    V_blur = apply_blockwise_blur_same_over_time(
        x_tchw=V,
        sigma_map=sigma_map_uv,
        block_h=clip_uv,
        block_w=clip_uv,
    )

    # Back to integer domain
    if args.bit_depth == 8:
        out_dtype = np.uint8
    else:
        out_dtype = np.uint16

    Y_out = torch.round(Y_blur.clamp(0, 1) * max_val).squeeze(1).cpu().numpy().astype(out_dtype)
    U_out = torch.round(U_blur.clamp(0, 1) * max_val).squeeze(1).cpu().numpy().astype(out_dtype)
    V_out = torch.round(V_blur.clamp(0, 1) * max_val).squeeze(1).cpu().numpy().astype(out_dtype)

    print(f"[Info] Writing blurred clip: {out_blur}")
    write_yuv420(str(out_blur), Y_out, U_out, V_out, args.bit_depth)

    print("[Done]")
    print(f"Original: {out_orig}")
    print(f"Blurred : {out_blur}")


if __name__ == "__main__":
    main()
