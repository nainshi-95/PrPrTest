# MCTF 1


import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MV9 import get_flow_three_level, load_images, pad_to_multiple, rgb_to_y_bt709, warp_8tap


def ensure_5d_y(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expected 4D or 5D Y tensor, got shape {tuple(x.shape)}")


def calculate_mctf_params(tar_y, ref_y, BS=16):
    """
    tar_y: (B, 1, 1, H, W) or (B, 1, H, W)
    ref_y: (B, 8, 1, H, W)
    BS: block size
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    tar_blks = tar.view(B, 1, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    ref_blks = ref.view(B, Ref_num, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    offset = 5.0
    scale = 50.0
    cntV = BS * BS
    cntD = 2 * cntV - BS - BS

    tar_avg = torch.mean(tar_blks, dim=(4, 5), keepdim=True)
    tar_var = torch.sum((tar_blks - tar_avg) ** 2, dim=(4, 5))

    diff_blks = tar_blks - ref_blks
    ssd = torch.sum(diff_blks ** 2, dim=(4, 5))

    error = (20.0 * (ssd + offset) / (tar_var + offset)) + ((ssd / cntV) / scale)

    diff_h = (diff_blks[..., :, 1:] - diff_blks[..., :, :-1]) ** 2
    diff_v = (diff_blks[..., 1:, :] - diff_blks[..., :-1, :]) ** 2
    diffsum = torch.sum(diff_h, dim=(4, 5)) + torch.sum(diff_v, dim=(4, 5))

    noise = torch.round((15.0 * (cntD / cntV) * ssd + offset) / (diffsum + offset))

    min_error, _ = torch.min(error, dim=1, keepdim=True)

    ww = torch.ones_like(error)
    sw = torch.ones_like(error)

    ww = torch.where(noise < 25, ww * 1.0, ww * 0.6)
    sw = torch.where(noise < 25, sw * 1.0, sw * 0.8)

    ww = torch.where(error < 50, ww * 1.2, torch.where(error > 100, ww * 0.6, ww))
    sw = torch.where(error < 50, sw * 1.0, sw * 0.8)

    ww = ww * ((min_error + 1.0) / (error + 1.0))

    return noise, error, ww, sw


def compute_actual_blending_weights(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    ww: torch.Tensor,
    sw: torch.Tensor,
    qp: int = 22,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size

    overall_strength = 1.0
    weight_scaling = overall_strength * 0.4
    sigma_zero_point = 10.0
    sigma_multiplier = 9.0
    sigma_sq = (qp - sigma_zero_point) ** 2 * sigma_multiplier

    ref_strengths = [0.85, 0.57, 0.41, 0.33]
    poc_offsets = [-4, -3, -2, -1, 1, 2, 3, 4]

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    ww_pix = torch.repeat_interleave(torch.repeat_interleave(ww, BS, dim=2), BS, dim=3).unsqueeze(2)
    sw_pix = torch.repeat_interleave(torch.repeat_interleave(sw, BS, dim=2), BS, dim=3).unsqueeze(2)

    ref_weight_raw = torch.zeros((B, Ref_num, 1, H, W), device=tar.device, dtype=tar.dtype)
    temporal_weight_sum = torch.ones((B, 1, 1, H, W), device=tar.device, dtype=tar.dtype)

    for i in range(Ref_num):
        diff = ref[:, i : i + 1] - tar
        diff_sq = diff ** 2

        dist_idx = min(3, abs(poc_offsets[i]) - 1)
        ref_str = ref_strengths[dist_idx]

        exponent = -diff_sq / (2.0 * sw_pix[:, i : i + 1] * sigma_sq)
        weight = weight_scaling * ref_str * ww_pix[:, i : i + 1] * torch.exp(exponent)

        ref_weight_raw[:, i : i + 1] = weight
        temporal_weight_sum += weight

    target_weight = 1.0 / temporal_weight_sum
    ref_weight = ref_weight_raw / temporal_weight_sum
    ref_weight_blk = F.avg_pool2d(
        ref_weight.squeeze(2).reshape(B * Ref_num, 1, H, W),
        kernel_size=BS,
        stride=BS,
    ).reshape(B, Ref_num, H // BS, W // BS)

    return target_weight, ref_weight, ref_weight_blk


def create_eight_warped_references(reference_y: torch.Tensor, flow: torch.Tensor, tile_rows: int = 256) -> torch.Tensor:
    B, _, H, W = reference_y.shape
    offsets = torch.tensor(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, -0.5],
            [0.0, 0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=flow.dtype,
        device=flow.device,
    )

    ref_rep = reference_y.unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(B * 8, 1, H, W)
    flow_rep = flow.unsqueeze(1).expand(-1, 8, -1, -1, -1).clone()
    flow_rep[:, :, 0, :, :] += offsets[:, 0].view(1, 8, 1, 1)
    flow_rep[:, :, 1, :, :] += offsets[:, 1].view(1, 8, 1, 1)
    flow_rep = flow_rep.reshape(B * 8, 2, H, W)

    warped = warp_8tap(ref_rep, flow_rep, tile_rows=tile_rows)
    return warped.view(B, 8, 1, H, W)


def compensate_reference_blocks(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    weight_map: torch.Tensor,
    weight_offset: float,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size
    HB = H // BS
    WB = W // BS

    tar_blks = tar_y.view(B, 1, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    ref_blks = ref_y.view(B, Ref_num, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()

    tar_mean = tar_blks.mean(dim=(-2, -1), keepdim=True)
    ref_mean = ref_blks.mean(dim=(-2, -1), keepdim=True)
    delta = tar_mean - ref_mean

    mask = (weight_map > weight_offset).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
    compensated_blks = torch.where(mask, ref_blks + delta, ref_blks)

    compensated = (
        compensated_blks.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, Ref_num, 1, H, W)
    )
    return torch.clamp(compensated, 0.0, 1.0), mask.squeeze(2).squeeze(-1).squeeze(-1)


def blend_with_block_weights(ref_y: torch.Tensor, weight_map: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, _, _ = ref_y.shape
    BS = block_size

    norm = weight_map.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weight_map / norm
    weights_pix = torch.repeat_interleave(torch.repeat_interleave(weights, BS, dim=2), BS, dim=3).unsqueeze(2)

    blended = (ref_y * weights_pix).sum(dim=1)
    return torch.clamp(blended, 0.0, 1.0)


def blend_with_actual_weights(
    tar_y: torch.Tensor, ref_y: torch.Tensor, target_weight: torch.Tensor, ref_weight: torch.Tensor
) -> torch.Tensor:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    tar_contrib = (tar_y * target_weight).sum(dim=1)
    ref_contrib = (ref_y * ref_weight).sum(dim=1)
    blended = tar_contrib + ref_contrib
    return torch.clamp(blended, 0.0, 1.0)


def save_y_png(y: torch.Tensor, out_path: str) -> None:
    if y.dim() == 4:
        img = y[0, 0]
    elif y.dim() == 3:
        img = y[0]
    elif y.dim() == 2:
        img = y
    else:
        raise ValueError(f"Unsupported image shape: {tuple(y.shape)}")

    arr = torch.clamp(img, 0.0, 1.0).detach().cpu().numpy()
    arr_u8 = np.round(arr * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


def save_diff_png(a: torch.Tensor, b: torch.Tensor, out_path: str) -> None:
    diff = torch.abs(a - b)
    max_val = diff.max().item()
    if max_val > 1e-8:
        diff = diff / max_val
    save_y_png(diff, out_path)


def compute_psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((max_val ** 2) / mse)


def main():
    parser = argparse.ArgumentParser(description="Block-wise MCTF blend demo with optional reference compensation")
    parser.add_argument("--target", type=str, default="1.png")
    parser.add_argument("--reference", type=str, default="2.png")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--sr", type=int, default=11)
    parser.add_argument("--chunk-k", type=int, default=16)
    parser.add_argument("--warp-tile-rows", type=int, default=256)
    parser.add_argument("--weight-offset", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.target).exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")
    if not Path(args.reference).exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rgb = load_images([args.target]).to(args.device)
    reference_rgb = load_images([args.reference]).to(args.device)

    if target_rgb.shape != reference_rgb.shape:
        raise ValueError("target and reference must have identical shape")

    multiple = args.bs * 4
    target_rgb_pad, orig_hw = pad_to_multiple(target_rgb, multiple)
    reference_rgb_pad, _ = pad_to_multiple(reference_rgb, multiple)

    target_y_pad = rgb_to_y_bt709(target_rgb_pad)
    reference_y_pad = rgb_to_y_bt709(reference_rgb_pad)

    _, _, flow_l0_int = get_flow_three_level(
        target_y_pad, reference_y_pad, bs=args.bs, sr=args.sr, chunk_k=args.chunk_k
    )

    warped_base_y_pad = warp_8tap(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)
    warped_refs_y_pad = create_eight_warped_references(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)

    tar_y_5d = target_y_pad.unsqueeze(1)
    noise, error, ww, sw = calculate_mctf_params(tar_y_5d, warped_refs_y_pad, BS=args.bs)
    target_weight_pad, ref_weight_pad, ref_weight_blk = compute_actual_blending_weights(
        tar_y_5d,
        warped_refs_y_pad,
        ww,
        sw,
        block_size=args.bs,
    )
    compensated_refs_y_pad, comp_mask = compensate_reference_blocks(
        tar_y_5d,
        warped_refs_y_pad,
        ref_weight_blk,
        weight_offset=args.weight_offset,
        block_size=args.bs,
    )

    blended_no_comp_pad = blend_with_actual_weights(tar_y_5d, warped_refs_y_pad, target_weight_pad, ref_weight_pad)
    blended_with_comp_pad = blend_with_actual_weights(
        tar_y_5d, compensated_refs_y_pad, target_weight_pad, ref_weight_pad
    )

    h0, w0 = orig_hw
    target_y = target_y_pad[:, :, :h0, :w0]
    warped_base_y = warped_base_y_pad[:, :, :h0, :w0]
    blended_no_comp = blended_no_comp_pad[:, :, :h0, :w0]
    blended_with_comp = blended_with_comp_pad[:, :, :h0, :w0]
    comp_delta = torch.abs(blended_with_comp - blended_no_comp)

    save_y_png(target_y, str(out_dir / "mctf_target_y.png"))
    save_y_png(warped_base_y, str(out_dir / "mctf_warped_base_y.png"))
    save_y_png(blended_no_comp, str(out_dir / "mctf_blended_no_comp_y.png"))
    save_y_png(blended_with_comp, str(out_dir / "mctf_blended_with_comp_y.png"))
    save_diff_png(blended_with_comp, blended_no_comp, str(out_dir / "mctf_blend_difference_y.png"))
    save_diff_png(target_y, blended_no_comp, str(out_dir / "mctf_error_no_comp_y.png"))
    save_diff_png(target_y, blended_with_comp, str(out_dir / "mctf_error_with_comp_y.png"))

    changed_blocks = int(comp_mask.sum().item())
    total_blocks = int(comp_mask.numel())
    psnr_no_comp = compute_psnr(target_y, blended_no_comp)
    psnr_with_comp = compute_psnr(target_y, blended_with_comp)

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"Target Y: {out_dir / 'mctf_target_y.png'}")
    print(f"Warped base Y: {out_dir / 'mctf_warped_base_y.png'}")
    print(f"Blended without compensation: {out_dir / 'mctf_blended_no_comp_y.png'}")
    print(f"Blended with compensation: {out_dir / 'mctf_blended_with_comp_y.png'}")
    print(f"Blend difference map: {out_dir / 'mctf_blend_difference_y.png'}")
    print(f"Error map without compensation: {out_dir / 'mctf_error_no_comp_y.png'}")
    print(f"Error map with compensation: {out_dir / 'mctf_error_with_comp_y.png'}")
    print(f"Compensated blocks: {changed_blocks} / {total_blocks}")
    print(f"PSNR without compensation: {psnr_no_comp:.6f} dB")
    print(f"PSNR with compensation: {psnr_with_comp:.6f} dB")
    print(f"Mean absolute blend delta: {comp_delta.mean().item():.8f}")


if __name__ == "__main__":
    main()














# MCTF 2
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MV9 import get_flow_three_level, load_images, pad_to_multiple, rgb_to_y_bt709, warp_8tap


def ensure_5d_y(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expected 4D or 5D Y tensor, got shape {tuple(x.shape)}")


def calculate_mctf_params(tar_y, ref_y, BS=16):
    """
    tar_y: (B, 1, 1, H, W) or (B, 1, H, W)
    ref_y: (B, 8, 1, H, W)
    BS: block size
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    tar_blks = tar.view(B, 1, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    ref_blks = ref.view(B, Ref_num, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    offset = 5.0
    scale = 50.0
    cntV = BS * BS
    cntD = 2 * cntV - BS - BS

    tar_avg = torch.mean(tar_blks, dim=(4, 5), keepdim=True)
    tar_var = torch.sum((tar_blks - tar_avg) ** 2, dim=(4, 5))

    diff_blks = tar_blks - ref_blks
    ssd = torch.sum(diff_blks ** 2, dim=(4, 5))

    error = (20.0 * (ssd + offset) / (tar_var + offset)) + ((ssd / cntV) / scale)

    diff_h = (diff_blks[..., :, 1:] - diff_blks[..., :, :-1]) ** 2
    diff_v = (diff_blks[..., 1:, :] - diff_blks[..., :-1, :]) ** 2
    diffsum = torch.sum(diff_h, dim=(4, 5)) + torch.sum(diff_v, dim=(4, 5))

    noise = torch.round((15.0 * (cntD / cntV) * ssd + offset) / (diffsum + offset))

    min_error, _ = torch.min(error, dim=1, keepdim=True)

    ww = torch.ones_like(error)
    sw = torch.ones_like(error)

    ww = torch.where(noise < 25, ww * 1.0, ww * 0.6)
    sw = torch.where(noise < 25, sw * 1.0, sw * 0.8)

    ww = torch.where(error < 50, ww * 1.2, torch.where(error > 100, ww * 0.6, ww))
    sw = torch.where(error < 50, sw * 1.0, sw * 0.8)

    ww = ww * ((min_error + 1.0) / (error + 1.0))

    return noise, error, ww, sw


def compute_actual_blending_weights(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    ww: torch.Tensor,
    sw: torch.Tensor,
    qp: int = 22,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size

    overall_strength = 1.0
    weight_scaling = overall_strength * 0.4
    sigma_zero_point = 10.0
    sigma_multiplier = 9.0
    sigma_sq = (qp - sigma_zero_point) ** 2 * sigma_multiplier

    ref_strengths = [0.85, 0.57, 0.41, 0.33]
    poc_offsets = [-4, -3, -2, -1, 1, 2, 3, 4]

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    ww_pix = torch.repeat_interleave(torch.repeat_interleave(ww, BS, dim=2), BS, dim=3).unsqueeze(2)
    sw_pix = torch.repeat_interleave(torch.repeat_interleave(sw, BS, dim=2), BS, dim=3).unsqueeze(2)

    ref_weight_raw = torch.zeros((B, Ref_num, 1, H, W), device=tar.device, dtype=tar.dtype)
    temporal_weight_sum = torch.ones((B, 1, 1, H, W), device=tar.device, dtype=tar.dtype)

    for i in range(Ref_num):
        diff = ref[:, i : i + 1] - tar
        diff_sq = diff ** 2

        dist_idx = min(3, abs(poc_offsets[i]) - 1)
        ref_str = ref_strengths[dist_idx]

        exponent = -diff_sq / (2.0 * sw_pix[:, i : i + 1] * sigma_sq)
        weight = weight_scaling * ref_str * ww_pix[:, i : i + 1] * torch.exp(exponent)

        ref_weight_raw[:, i : i + 1] = weight
        temporal_weight_sum += weight

    target_weight = 1.0 / temporal_weight_sum
    ref_weight = ref_weight_raw / temporal_weight_sum
    ref_weight_blk = F.avg_pool2d(
        ref_weight.squeeze(2).reshape(B * Ref_num, 1, H, W),
        kernel_size=BS,
        stride=BS,
    ).reshape(B, Ref_num, H // BS, W // BS)

    return target_weight, ref_weight, ref_weight_blk


def create_eight_warped_references(reference_y: torch.Tensor, flow: torch.Tensor, tile_rows: int = 256) -> torch.Tensor:
    B, _, H, W = reference_y.shape
    offsets = torch.tensor(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, -0.5],
            [0.0, 0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=flow.dtype,
        device=flow.device,
    )

    ref_rep = reference_y.unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(B * 8, 1, H, W)
    flow_rep = flow.unsqueeze(1).expand(-1, 8, -1, -1, -1).clone()
    flow_rep[:, :, 0, :, :] += offsets[:, 0].view(1, 8, 1, 1)
    flow_rep[:, :, 1, :, :] += offsets[:, 1].view(1, 8, 1, 1)
    flow_rep = flow_rep.reshape(B * 8, 2, H, W)

    warped = warp_8tap(ref_rep, flow_rep, tile_rows=tile_rows)
    return warped.view(B, 8, 1, H, W)


def compensate_reference_blocks(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size
    HB = H // BS
    WB = W // BS

    tar_blks = tar_y.view(B, 1, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    ref_blks = ref_y.view(B, Ref_num, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()

    tar_mean = tar_blks.mean(dim=(-2, -1), keepdim=True)
    ref_mean = ref_blks.mean(dim=(-2, -1), keepdim=True)
    delta = tar_mean - ref_mean

    compensated_blks = ref_blks + delta

    compensated = (
        compensated_blks.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, Ref_num, 1, H, W)
    )
    return torch.clamp(compensated, 0.0, 1.0)


def blend_with_block_weights(ref_y: torch.Tensor, weight_map: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, _, _ = ref_y.shape
    BS = block_size

    norm = weight_map.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weight_map / norm
    weights_pix = torch.repeat_interleave(torch.repeat_interleave(weights, BS, dim=2), BS, dim=3).unsqueeze(2)

    blended = (ref_y * weights_pix).sum(dim=1)
    return torch.clamp(blended, 0.0, 1.0)


def blend_with_actual_weights(
    tar_y: torch.Tensor, ref_y: torch.Tensor, target_weight: torch.Tensor, ref_weight: torch.Tensor
) -> torch.Tensor:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    tar_contrib = (tar_y * target_weight).sum(dim=1)
    ref_contrib = (ref_y * ref_weight).sum(dim=1)
    blended = tar_contrib + ref_contrib
    return torch.clamp(blended, 0.0, 1.0)


def select_better_compensated_blocks(
    target_y: torch.Tensor,
    blended_no_comp: torch.Tensor,
    blended_with_comp: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    BS = block_size
    B, C, H, W = target_y.shape
    HB = H // BS
    WB = W // BS

    tar_blks = target_y.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    no_comp_blks = blended_no_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    with_comp_blks = blended_with_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    mse_no_comp = torch.mean((tar_blks - no_comp_blks) ** 2, dim=(1, 4, 5))
    mse_with_comp = torch.mean((tar_blks - with_comp_blks) ** 2, dim=(1, 4, 5))
    better_mask = mse_with_comp < mse_no_comp

    block_mask = better_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    selected_blks = torch.where(block_mask, with_comp_blks, no_comp_blks)
    selected = selected_blks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

    return selected, better_mask


def save_y_png(y: torch.Tensor, out_path: str) -> None:
    if y.dim() == 4:
        img = y[0, 0]
    elif y.dim() == 3:
        img = y[0]
    elif y.dim() == 2:
        img = y
    else:
        raise ValueError(f"Unsupported image shape: {tuple(y.shape)}")

    arr = torch.clamp(img, 0.0, 1.0).detach().cpu().numpy()
    arr_u8 = np.round(arr * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


def save_diff_png(a: torch.Tensor, b: torch.Tensor, out_path: str) -> None:
    diff = torch.abs(a - b)
    max_val = diff.max().item()
    if max_val > 1e-8:
        diff = diff / max_val
    save_y_png(diff, out_path)


def compute_psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((max_val ** 2) / mse)


def main():
    parser = argparse.ArgumentParser(description="Block-wise MCTF blend demo with optional reference compensation")
    parser.add_argument("--target", type=str, default="1.png")
    parser.add_argument("--reference", type=str, default="2.png")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--sr", type=int, default=11)
    parser.add_argument("--chunk-k", type=int, default=16)
    parser.add_argument("--warp-tile-rows", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.target).exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")
    if not Path(args.reference).exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rgb = load_images([args.target]).to(args.device)
    reference_rgb = load_images([args.reference]).to(args.device)

    if target_rgb.shape != reference_rgb.shape:
        raise ValueError("target and reference must have identical shape")

    multiple = args.bs * 4
    target_rgb_pad, orig_hw = pad_to_multiple(target_rgb, multiple)
    reference_rgb_pad, _ = pad_to_multiple(reference_rgb, multiple)

    target_y_pad = rgb_to_y_bt709(target_rgb_pad)
    reference_y_pad = rgb_to_y_bt709(reference_rgb_pad)

    _, _, flow_l0_int = get_flow_three_level(
        target_y_pad, reference_y_pad, bs=args.bs, sr=args.sr, chunk_k=args.chunk_k
    )

    warped_base_y_pad = warp_8tap(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)
    warped_refs_y_pad = create_eight_warped_references(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)

    tar_y_5d = target_y_pad.unsqueeze(1)
    noise, error, ww, sw = calculate_mctf_params(tar_y_5d, warped_refs_y_pad, BS=args.bs)
    target_weight_pad, ref_weight_pad, ref_weight_blk = compute_actual_blending_weights(
        tar_y_5d,
        warped_refs_y_pad,
        ww,
        sw,
        block_size=args.bs,
    )
    compensated_refs_y_pad = compensate_reference_blocks(tar_y_5d, warped_refs_y_pad, block_size=args.bs)

    blended_no_comp_pad = blend_with_actual_weights(tar_y_5d, warped_refs_y_pad, target_weight_pad, ref_weight_pad)
    blended_all_comp_pad = blend_with_actual_weights(
        tar_y_5d, compensated_refs_y_pad, target_weight_pad, ref_weight_pad
    )
    blended_with_comp_pad, better_block_mask = select_better_compensated_blocks(
        target_y_pad,
        blended_no_comp_pad,
        blended_all_comp_pad,
        block_size=args.bs,
    )

    h0, w0 = orig_hw
    target_y = target_y_pad[:, :, :h0, :w0]
    warped_base_y = warped_base_y_pad[:, :, :h0, :w0]
    blended_no_comp = blended_no_comp_pad[:, :, :h0, :w0]
    blended_with_comp = blended_with_comp_pad[:, :, :h0, :w0]
    comp_delta = torch.abs(blended_with_comp - blended_no_comp)

    save_y_png(target_y, str(out_dir / "mctf_target_y.png"))
    save_y_png(warped_base_y, str(out_dir / "mctf_warped_base_y.png"))
    save_y_png(blended_no_comp, str(out_dir / "mctf_blended_no_comp_y.png"))
    save_y_png(blended_with_comp, str(out_dir / "mctf_blended_with_comp_y.png"))
    save_diff_png(blended_with_comp, blended_no_comp, str(out_dir / "mctf_blend_difference_y.png"))
    save_diff_png(target_y, blended_no_comp, str(out_dir / "mctf_error_no_comp_y.png"))
    save_diff_png(target_y, blended_with_comp, str(out_dir / "mctf_error_with_comp_y.png"))

    changed_blocks = int(better_block_mask.sum().item())
    total_blocks = int(better_block_mask.numel())
    psnr_no_comp = compute_psnr(target_y, blended_no_comp)
    psnr_with_comp = compute_psnr(target_y, blended_with_comp)

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"Target Y: {out_dir / 'mctf_target_y.png'}")
    print(f"Warped base Y: {out_dir / 'mctf_warped_base_y.png'}")
    print(f"Blended without compensation: {out_dir / 'mctf_blended_no_comp_y.png'}")
    print(f"Blended with compensation: {out_dir / 'mctf_blended_with_comp_y.png'}")
    print(f"Blend difference map: {out_dir / 'mctf_blend_difference_y.png'}")
    print(f"Error map without compensation: {out_dir / 'mctf_error_no_comp_y.png'}")
    print(f"Error map with compensation: {out_dir / 'mctf_error_with_comp_y.png'}")
    print(f"Compensated blocks: {changed_blocks} / {total_blocks}")
    print(f"PSNR without compensation: {psnr_no_comp:.6f} dB")
    print(f"PSNR with compensation: {psnr_with_comp:.6f} dB")
    print(f"Mean reference block weight: {ref_weight_blk.mean().item():.6f}")
    print(f"Mean absolute blend delta: {comp_delta.mean().item():.8f}")


if __name__ == "__main__":
    main()

















# MCTF 3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MV9 import get_flow_three_level, load_images, pad_to_multiple, rgb_to_y_bt709, warp_8tap


def ensure_5d_y(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expected 4D or 5D Y tensor, got shape {tuple(x.shape)}")


def calculate_mctf_params(tar_y, ref_y, BS=16):
    """
    tar_y: (B, 1, 1, H, W) or (B, 1, H, W)
    ref_y: (B, 8, 1, H, W)
    BS: block size
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    tar_blks = tar.view(B, 1, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    ref_blks = ref.view(B, Ref_num, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    offset = 5.0
    scale = 50.0
    cntV = BS * BS
    cntD = 2 * cntV - BS - BS

    tar_avg = torch.mean(tar_blks, dim=(4, 5), keepdim=True)
    tar_var = torch.sum((tar_blks - tar_avg) ** 2, dim=(4, 5))

    diff_blks = tar_blks - ref_blks
    ssd = torch.sum(diff_blks ** 2, dim=(4, 5))

    error = (20.0 * (ssd + offset) / (tar_var + offset)) + ((ssd / cntV) / scale)

    diff_h = (diff_blks[..., :, 1:] - diff_blks[..., :, :-1]) ** 2
    diff_v = (diff_blks[..., 1:, :] - diff_blks[..., :-1, :]) ** 2
    diffsum = torch.sum(diff_h, dim=(4, 5)) + torch.sum(diff_v, dim=(4, 5))

    noise = torch.round((15.0 * (cntD / cntV) * ssd + offset) / (diffsum + offset))

    min_error, _ = torch.min(error, dim=1, keepdim=True)

    ww = torch.ones_like(error)
    sw = torch.ones_like(error)

    ww = torch.where(noise < 25, ww * 1.0, ww * 0.6)
    sw = torch.where(noise < 25, sw * 1.0, sw * 0.8)

    ww = torch.where(error < 50, ww * 1.2, torch.where(error > 100, ww * 0.6, ww))
    sw = torch.where(error < 50, sw * 1.0, sw * 0.8)

    ww = ww * ((min_error + 1.0) / (error + 1.0))

    return noise, error, ww, sw


def compute_actual_blending_weights(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    ww: torch.Tensor,
    sw: torch.Tensor,
    qp: int = 22,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size

    overall_strength = 1.0
    weight_scaling = overall_strength * 0.4
    sigma_zero_point = 10.0
    sigma_multiplier = 9.0
    sigma_sq = (qp - sigma_zero_point) ** 2 * sigma_multiplier

    ref_strengths = [0.85, 0.57, 0.41, 0.33]
    poc_offsets = [-4, -3, -2, -1, 1, 2, 3, 4]

    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    ww_pix = torch.repeat_interleave(torch.repeat_interleave(ww, BS, dim=2), BS, dim=3).unsqueeze(2)
    sw_pix = torch.repeat_interleave(torch.repeat_interleave(sw, BS, dim=2), BS, dim=3).unsqueeze(2)

    ref_weight_raw = torch.zeros((B, Ref_num, 1, H, W), device=tar.device, dtype=tar.dtype)
    temporal_weight_sum = torch.ones((B, 1, 1, H, W), device=tar.device, dtype=tar.dtype)

    for i in range(Ref_num):
        diff = ref[:, i : i + 1] - tar
        diff_sq = diff ** 2

        dist_idx = min(3, abs(poc_offsets[i]) - 1)
        ref_str = ref_strengths[dist_idx]

        exponent = -diff_sq / (2.0 * sw_pix[:, i : i + 1] * sigma_sq)
        weight = weight_scaling * ref_str * ww_pix[:, i : i + 1] * torch.exp(exponent)

        ref_weight_raw[:, i : i + 1] = weight
        temporal_weight_sum += weight

    target_weight = 1.0 / temporal_weight_sum
    ref_weight = ref_weight_raw / temporal_weight_sum
    ref_weight_blk = F.avg_pool2d(
        ref_weight.squeeze(2).reshape(B * Ref_num, 1, H, W),
        kernel_size=BS,
        stride=BS,
    ).reshape(B, Ref_num, H // BS, W // BS)

    return target_weight, ref_weight, ref_weight_blk


def create_eight_warped_references(reference_y: torch.Tensor, flow: torch.Tensor, tile_rows: int = 256) -> torch.Tensor:
    B, _, H, W = reference_y.shape
    offsets = torch.tensor(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, -0.5],
            [0.0, 0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=flow.dtype,
        device=flow.device,
    )

    ref_rep = reference_y.unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(B * 8, 1, H, W)
    flow_rep = flow.unsqueeze(1).expand(-1, 8, -1, -1, -1).clone()
    flow_rep[:, :, 0, :, :] += offsets[:, 0].view(1, 8, 1, 1)
    flow_rep[:, :, 1, :, :] += offsets[:, 1].view(1, 8, 1, 1)
    flow_rep = flow_rep.reshape(B * 8, 2, H, W)

    warped = warp_8tap(ref_rep, flow_rep, tile_rows=tile_rows)
    return warped.view(B, 8, 1, H, W)


def compensate_reference_blocks(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    weight_map: torch.Tensor,
    weight_offset: float,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size
    HB = H // BS
    WB = W // BS

    tar_blks = tar_y.view(B, 1, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    ref_blks = ref_y.view(B, Ref_num, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()

    tar_mean = tar_blks.mean(dim=(-2, -1), keepdim=True)
    ref_mean = ref_blks.mean(dim=(-2, -1), keepdim=True)
    delta = tar_mean - ref_mean

    mask = (weight_map > weight_offset).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
    compensated_blks = torch.where(mask, ref_blks + delta, ref_blks)

    compensated = (
        compensated_blks.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, Ref_num, 1, H, W)
    )
    return torch.clamp(compensated, 0.0, 1.0), mask.squeeze(2).squeeze(-1).squeeze(-1)


def blend_with_block_weights(ref_y: torch.Tensor, weight_map: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, _, _ = ref_y.shape
    BS = block_size

    norm = weight_map.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weight_map / norm
    weights_pix = torch.repeat_interleave(torch.repeat_interleave(weights, BS, dim=2), BS, dim=3).unsqueeze(2)

    blended = (ref_y * weights_pix).sum(dim=1)
    return torch.clamp(blended, 0.0, 1.0)


def blend_with_actual_weights(
    tar_y: torch.Tensor, ref_y: torch.Tensor, target_weight: torch.Tensor, ref_weight: torch.Tensor
) -> torch.Tensor:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    tar_contrib = (tar_y * target_weight).sum(dim=1)
    ref_contrib = (ref_y * ref_weight).sum(dim=1)
    blended = tar_contrib + ref_contrib
    return torch.clamp(blended, 0.0, 1.0)


def save_y_png(y: torch.Tensor, out_path: str) -> None:
    if y.dim() == 4:
        img = y[0, 0]
    elif y.dim() == 3:
        img = y[0]
    elif y.dim() == 2:
        img = y
    else:
        raise ValueError(f"Unsupported image shape: {tuple(y.shape)}")

    arr = torch.clamp(img, 0.0, 1.0).detach().cpu().numpy()
    arr_u8 = np.round(arr * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


def save_diff_png(a: torch.Tensor, b: torch.Tensor, out_path: str) -> None:
    diff = torch.abs(a - b)
    max_val = diff.max().item()
    if max_val > 1e-8:
        diff = diff / max_val
    save_y_png(diff, out_path)


def compute_psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((max_val ** 2) / mse)


def main():
    parser = argparse.ArgumentParser(description="Block-wise MCTF blend demo with optional reference compensation")
    parser.add_argument("--target", type=str, default="1.png")
    parser.add_argument("--reference", type=str, default="2.png")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--sr", type=int, default=11)
    parser.add_argument("--chunk-k", type=int, default=16)
    parser.add_argument("--warp-tile-rows", type=int, default=256)
    parser.add_argument("--weight-offset", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.target).exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")
    if not Path(args.reference).exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rgb = load_images([args.target]).to(args.device)
    reference_rgb = load_images([args.reference]).to(args.device)

    if target_rgb.shape != reference_rgb.shape:
        raise ValueError("target and reference must have identical shape")

    multiple = args.bs * 4
    target_rgb_pad, orig_hw = pad_to_multiple(target_rgb, multiple)
    reference_rgb_pad, _ = pad_to_multiple(reference_rgb, multiple)

    target_y_pad = rgb_to_y_bt709(target_rgb_pad)
    reference_y_pad = rgb_to_y_bt709(reference_rgb_pad)

    _, _, flow_l0_int = get_flow_three_level(
        target_y_pad, reference_y_pad, bs=args.bs, sr=args.sr, chunk_k=args.chunk_k
    )

    warped_base_y_pad = warp_8tap(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)
    warped_refs_y_pad = create_eight_warped_references(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)

    tar_y_5d = target_y_pad.unsqueeze(1)
    noise, error, ww, sw = calculate_mctf_params(tar_y_5d, warped_refs_y_pad, BS=args.bs)
    target_weight_pad, ref_weight_pad, ref_weight_blk = compute_actual_blending_weights(
        tar_y_5d,
        warped_refs_y_pad,
        ww,
        sw,
        block_size=args.bs,
    )
    compensated_refs_y_pad, comp_mask = compensate_reference_blocks(
        tar_y_5d,
        warped_refs_y_pad,
        ref_weight_blk,
        weight_offset=args.weight_offset,
        block_size=args.bs,
    )

    blended_no_comp_pad = blend_with_actual_weights(tar_y_5d, warped_refs_y_pad, target_weight_pad, ref_weight_pad)
    target_weight_comp_pad, ref_weight_comp_pad, _ = compute_actual_blending_weights(
        tar_y_5d,
        compensated_refs_y_pad,
        ww,
        sw,
        block_size=args.bs,
    )
    blended_with_comp_pad = blend_with_actual_weights(
        tar_y_5d, compensated_refs_y_pad, target_weight_comp_pad, ref_weight_comp_pad
    )

    h0, w0 = orig_hw
    target_y = target_y_pad[:, :, :h0, :w0]
    warped_base_y = warped_base_y_pad[:, :, :h0, :w0]
    blended_no_comp = blended_no_comp_pad[:, :, :h0, :w0]
    blended_with_comp = blended_with_comp_pad[:, :, :h0, :w0]
    comp_delta = torch.abs(blended_with_comp - blended_no_comp)

    save_y_png(target_y, str(out_dir / "mctf_target_y.png"))
    save_y_png(warped_base_y, str(out_dir / "mctf_warped_base_y.png"))
    save_y_png(blended_no_comp, str(out_dir / "mctf_blended_no_comp_y.png"))
    save_y_png(blended_with_comp, str(out_dir / "mctf_blended_with_comp_y.png"))
    save_diff_png(blended_with_comp, blended_no_comp, str(out_dir / "mctf_blend_difference_y.png"))
    save_diff_png(target_y, blended_no_comp, str(out_dir / "mctf_error_no_comp_y.png"))
    save_diff_png(target_y, blended_with_comp, str(out_dir / "mctf_error_with_comp_y.png"))

    changed_blocks = int(comp_mask.sum().item())
    total_blocks = int(comp_mask.numel())
    psnr_no_comp = compute_psnr(target_y, blended_no_comp)
    psnr_with_comp = compute_psnr(target_y, blended_with_comp)

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"Target Y: {out_dir / 'mctf_target_y.png'}")
    print(f"Warped base Y: {out_dir / 'mctf_warped_base_y.png'}")
    print(f"Blended without compensation: {out_dir / 'mctf_blended_no_comp_y.png'}")
    print(f"Blended with compensation: {out_dir / 'mctf_blended_with_comp_y.png'}")
    print(f"Blend difference map: {out_dir / 'mctf_blend_difference_y.png'}")
    print(f"Error map without compensation: {out_dir / 'mctf_error_no_comp_y.png'}")
    print(f"Error map with compensation: {out_dir / 'mctf_error_with_comp_y.png'}")
    print(f"Compensated blocks: {changed_blocks} / {total_blocks}")
    print(f"PSNR without compensation: {psnr_no_comp:.6f} dB")
    print(f"PSNR with compensation: {psnr_with_comp:.6f} dB")


if __name__ == "__main__":
    main()

























# MCTF4
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MV9 import get_flow_three_level, load_images, pad_to_multiple, rgb_to_y_bt709, warp_8tap


# ============================================================
# Utils
# ============================================================
def ensure_5d_y(x: torch.Tensor) -> torch.Tensor:
    # Y: (B,1,H,W) or (B,Ref,1,H,W) -> ensure 5D
    if x.dim() == 4:
        return x.unsqueeze(1)  # (B,1,1,H,W)
    if x.dim() == 5:
        return x
    raise ValueError(f"Expected 4D or 5D Y tensor, got shape {tuple(x.shape)}")


def compute_psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((max_val ** 2) / mse)


def save_y_png(y: torch.Tensor, out_path: str) -> None:
    if y.dim() == 4:
        img = y[0, 0]
    elif y.dim() == 3:
        img = y[0]
    elif y.dim() == 2:
        img = y
    else:
        raise ValueError(f"Unsupported image shape: {tuple(y.shape)}")

    arr = torch.clamp(img, 0.0, 1.0).detach().cpu().numpy()
    arr_u8 = np.round(arr * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


def save_diff_png(a: torch.Tensor, b: torch.Tensor, out_path: str) -> None:
    diff = torch.abs(a - b)
    max_val = diff.max().item()
    if max_val > 1e-8:
        diff = diff / max_val
    save_y_png(diff, out_path)


# ============================================================
# 1) Original MCTF param calc (unchanged)
# ============================================================
def calculate_mctf_params(tar_y, ref_y, BS=16):
    """
    tar_y: (B, 1, 1, H, W) or (B, 1, H, W)
    ref_y: (B, 8, 1, H, W)
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    tar_blks = tar.view(B, 1, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    ref_blks = ref.view(B, Ref_num, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    offset = 5.0
    scale = 50.0
    cntV = BS * BS
    cntD = 2 * cntV - BS - BS

    tar_avg = torch.mean(tar_blks, dim=(4, 5), keepdim=True)
    tar_var = torch.sum((tar_blks - tar_avg) ** 2, dim=(4, 5))

    diff_blks = tar_blks - ref_blks
    ssd = torch.sum(diff_blks ** 2, dim=(4, 5))

    error = (20.0 * (ssd + offset) / (tar_var + offset)) + ((ssd / cntV) / scale)

    diff_h = (diff_blks[..., :, 1:] - diff_blks[..., :, :-1]) ** 2
    diff_v = (diff_blks[..., 1:, :] - diff_blks[..., :-1, :]) ** 2
    diffsum = torch.sum(diff_h, dim=(4, 5)) + torch.sum(diff_v, dim=(4, 5))

    noise = torch.round((15.0 * (cntD / cntV) * ssd + offset) / (diffsum + offset))

    min_error, _ = torch.min(error, dim=1, keepdim=True)

    ww = torch.ones_like(error)
    sw = torch.ones_like(error)

    ww = torch.where(noise < 25, ww * 1.0, ww * 0.6)
    sw = torch.where(noise < 25, sw * 1.0, sw * 0.8)

    ww = torch.where(error < 50, ww * 1.2, torch.where(error > 100, ww * 0.6, ww))
    sw = torch.where(error < 50, sw * 1.0, sw * 0.8)

    ww = ww * ((min_error + 1.0) / (error + 1.0))

    return noise, error, ww, sw


# ============================================================
# 2) Actual blending weights (option: use compensated refs ONLY for diff/exponent)
# ============================================================
def compute_actual_blending_weights(
    tar_y: torch.Tensor,
    ref_y_for_diff: torch.Tensor,   # used ONLY for diff/exponent
    ww: torch.Tensor,
    sw: torch.Tensor,
    qp: int = 22,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    tar_y: (B,1,H,W) or (B,1,1,H,W)
    ref_y_for_diff: (B,8,1,H,W)  (can be original refs or DC-comp refs)
    ww, sw: (B,8,HB,WB)
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y_for_diff = ensure_5d_y(ref_y_for_diff)

    B, Ref_num, _, H, W = ref_y_for_diff.shape
    BS = block_size

    overall_strength = 1.0
    weight_scaling = overall_strength * 0.4
    sigma_zero_point = 10.0
    sigma_multiplier = 9.0
    sigma_sq = (qp - sigma_zero_point) ** 2 * sigma_multiplier

    ref_strengths = [0.85, 0.57, 0.41, 0.33]
    poc_offsets = [-4, -3, -2, -1, 1, 2, 3, 4]

    tar = tar_y * 1023.0
    ref = ref_y_for_diff * 1023.0

    ww_pix = torch.repeat_interleave(torch.repeat_interleave(ww, BS, dim=2), BS, dim=3).unsqueeze(2)
    sw_pix = torch.repeat_interleave(torch.repeat_interleave(sw, BS, dim=2), BS, dim=3).unsqueeze(2)

    ref_weight_raw = torch.zeros((B, Ref_num, 1, H, W), device=tar.device, dtype=tar.dtype)
    temporal_weight_sum = torch.ones((B, 1, 1, H, W), device=tar.device, dtype=tar.dtype)

    for i in range(Ref_num):
        diff = ref[:, i : i + 1] - tar
        diff_sq = diff ** 2

        dist_idx = min(3, abs(poc_offsets[i]) - 1)
        ref_str = ref_strengths[dist_idx]

        exponent = -diff_sq / (2.0 * sw_pix[:, i : i + 1] * sigma_sq)
        weight = weight_scaling * ref_str * ww_pix[:, i : i + 1] * torch.exp(exponent)

        ref_weight_raw[:, i : i + 1] = weight
        temporal_weight_sum += weight

    target_weight = 1.0 / temporal_weight_sum
    ref_weight = ref_weight_raw / temporal_weight_sum

    # block-avg ref weight (for gating which refs to compensate)
    ref_weight_blk = F.avg_pool2d(
        ref_weight.squeeze(2).reshape(B * Ref_num, 1, H, W),
        kernel_size=BS,
        stride=BS,
    ).reshape(B, Ref_num, H // BS, W // BS)

    return target_weight, ref_weight, ref_weight_blk


# ============================================================
# 3) Create 8 warped refs (unchanged)
# ============================================================
def create_eight_warped_references(reference_y: torch.Tensor, flow: torch.Tensor, tile_rows: int = 256) -> torch.Tensor:
    B, _, H, W = reference_y.shape
    offsets = torch.tensor(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.0, -0.5],
            [0.0, 0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=flow.dtype,
        device=flow.device,
    )

    ref_rep = reference_y.unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(B * 8, 1, H, W)
    flow_rep = flow.unsqueeze(1).expand(-1, 8, -1, -1, -1).clone()
    flow_rep[:, :, 0, :, :] += offsets[:, 0].view(1, 8, 1, 1)
    flow_rep[:, :, 1, :, :] += offsets[:, 1].view(1, 8, 1, 1)
    flow_rep = flow_rep.reshape(B * 8, 2, H, W)

    warped = warp_8tap(ref_rep, flow_rep, tile_rows=tile_rows)
    return warped.view(B, 8, 1, H, W)


# ============================================================
# 4) DC compensation (modified: compensate only reliable refs per-block)
# ============================================================
def compensate_reference_blocks(
    tar_y: torch.Tensor,
    ref_y: torch.Tensor,
    ref_mask_blk: torch.Tensor | None = None,  # (B,Ref,HB,WB) bool/0-1
    block_size: int = 16,
    do_clamp: bool = True,
) -> torch.Tensor:
    """
    tar_y: (B,1,1,H,W) or (B,1,H,W)
    ref_y: (B,Ref,1,H,W)
    ref_mask_blk: optional block mask per-ref indicating which blocks to compensate.
                  1/True => apply DC compensation, 0/False => keep original.
    """
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    B, Ref_num, _, H, W = ref_y.shape
    BS = block_size
    HB = H // BS
    WB = W // BS

    tar_blks = tar_y.view(B, 1, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    ref_blks = ref_y.view(B, Ref_num, 1, HB, BS, WB, BS).permute(0, 1, 2, 3, 5, 4, 6).contiguous()

    tar_mean = tar_blks.mean(dim=(-2, -1), keepdim=True)
    ref_mean = ref_blks.mean(dim=(-2, -1), keepdim=True)
    delta = tar_mean - ref_mean  # (B,Ref,1,HB,WB,1,1) after broadcast

    compensated_blks = ref_blks + delta

    if ref_mask_blk is not None:
        # ref_mask_blk: (B,Ref,HB,WB) -> (B,Ref,1,HB,WB,1,1)
        m = ref_mask_blk.to(dtype=ref_blks.dtype, device=ref_blks.device)
        m = m.view(B, Ref_num, 1, HB, WB, 1, 1)
        compensated_blks = m * compensated_blks + (1.0 - m) * ref_blks

    compensated = compensated_blks.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, Ref_num, 1, H, W)
    if do_clamp:
        compensated = torch.clamp(compensated, 0.0, 1.0)
    return compensated


# ============================================================
# 5) Blend (unchanged)
# ============================================================
def blend_with_actual_weights(
    tar_y: torch.Tensor, ref_y: torch.Tensor, target_weight: torch.Tensor, ref_weight: torch.Tensor
) -> torch.Tensor:
    tar_y = ensure_5d_y(tar_y)
    ref_y = ensure_5d_y(ref_y)

    tar_contrib = (tar_y * target_weight).sum(dim=1)
    ref_contrib = (ref_y * ref_weight).sum(dim=1)
    blended = tar_contrib + ref_contrib
    return torch.clamp(blended, 0.0, 1.0)


# ============================================================
# 6) Soft gate selection (replaces hard block switch)
# ============================================================
def soft_gate_blocks(
    target_y: torch.Tensor,
    blended_no_comp: torch.Tensor,
    blended_with_comp: torch.Tensor,
    block_size: int = 16,
    tau: float = 0.0,
    slope: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute blockwise gate g in [0,1] from MSE improvement and blend smoothly.
    - tau: improvement threshold (0 means "any improvement")
    - slope: larger => closer to hard switch, smaller => smoother

    Returns:
      selected: (B,1,H,W)
      gate_blk: (B,HB,WB) in [0,1]
    """
    BS = block_size
    B, C, H, W = target_y.shape
    HB = H // BS
    WB = W // BS

    tar_blks = target_y.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    no_blks  = blended_no_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    co_blks  = blended_with_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    mse_no = torch.mean((tar_blks - no_blks) ** 2, dim=(1, 4, 5))      # (B,HB,WB)
    mse_co = torch.mean((tar_blks - co_blks) ** 2, dim=(1, 4, 5))      # (B,HB,WB)
    improve = mse_no - mse_co                                          # positive => comp better

    # gate: sigmoid(slope*(improve - tau) / (mse_no + eps))
    eps = 1e-12
    rel = (improve - tau) / (mse_no + eps)
    gate_blk = torch.sigmoid(slope * rel).clamp(0.0, 1.0)              # (B,HB,WB)

    gate_pix = torch.repeat_interleave(torch.repeat_interleave(gate_blk, BS, dim=1), BS, dim=2)  # (B,H,W)
    gate_pix = gate_pix.unsqueeze(1)                                   # (B,1,H,W)

    selected = (1.0 - gate_pix) * blended_no_comp + gate_pix * blended_with_comp
    selected = torch.clamp(selected, 0.0, 1.0)
    return selected, gate_blk


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MCTF blend demo with DC-comp refs + soft gating")
    parser.add_argument("--target", type=str, default="1.png")
    parser.add_argument("--reference", type=str, default="2.png")

    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--sr", type=int, default=11)
    parser.add_argument("--chunk-k", type=int, default=16)
    parser.add_argument("--warp-tile-rows", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default=".")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- New options ---
    parser.add_argument("--ref-comp-weight-thr", type=float, default=0.02,
                        help="Compensate DC only for refs whose block-avg ref weight >= this threshold.")
    parser.add_argument("--use-comp-in-weight", action="store_true",
                        help="Use DC-comp refs only for diff/exponent when computing weights (ww/sw/noise/error unchanged).")
    parser.add_argument("--soft-gate", action="store_true",
                        help="Use soft block gate instead of hard selection. Strongly recommended.")
    parser.add_argument("--gate-tau", type=float, default=0.0, help="Soft gate improvement threshold.")
    parser.add_argument("--gate-slope", type=float, default=6.0, help="Soft gate slope (bigger -> harder).")
    parser.add_argument("--no-clamp-comp", action="store_true",
                        help="Disable clamp on compensated refs (can reduce clipping artifacts; test carefully).")

    args = parser.parse_args()

    if not Path(args.target).exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")
    if not Path(args.reference).exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rgb = load_images([args.target]).to(args.device)
    reference_rgb = load_images([args.reference]).to(args.device)
    if target_rgb.shape != reference_rgb.shape:
        raise ValueError("target and reference must have identical shape")

    multiple = args.bs * 4
    target_rgb_pad, orig_hw = pad_to_multiple(target_rgb, multiple)
    reference_rgb_pad, _ = pad_to_multiple(reference_rgb, multiple)

    target_y_pad = rgb_to_y_bt709(target_rgb_pad)
    reference_y_pad = rgb_to_y_bt709(reference_rgb_pad)

    # Flow + warp
    _, _, flow_l0_int = get_flow_three_level(
        target_y_pad, reference_y_pad, bs=args.bs, sr=args.sr, chunk_k=args.chunk_k
    )

    warped_base_y_pad = warp_8tap(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)
    warped_refs_y_pad = create_eight_warped_references(reference_y_pad, flow_l0_int, tile_rows=args.warp_tile_rows)

    tar_y_5d = target_y_pad.unsqueeze(1)

    # MCTF params (unchanged)
    noise, error, ww, sw = calculate_mctf_params(tar_y_5d, warped_refs_y_pad, BS=args.bs)

    # ------------------------------------------------------------
    # First compute weights using ORIGINAL refs (baseline).
    # We will use ref_weight_blk to decide which refs to DC-comp per block.
    # ------------------------------------------------------------
    target_weight_pad_base, ref_weight_pad_base, ref_weight_blk_base = compute_actual_blending_weights(
        tar_y_5d,
        warped_refs_y_pad,  # diff from original
        ww,
        sw,
        block_size=args.bs,
    )

    # Decide which refs to compensate per-block based on baseline ref_weight_blk
    ref_mask_blk = (ref_weight_blk_base >= float(args.ref_comp_weight_thr))  # (B,Ref,HB,WB) bool

    compensated_refs_y_pad = compensate_reference_blocks(
        tar_y_5d,
        warped_refs_y_pad,
        ref_mask_blk=ref_mask_blk,
        block_size=args.bs,
        do_clamp=(not args.no_clamp_comp),
    )

    # ------------------------------------------------------------
    # Optionally recompute weights' diff/exponent using compensated refs,
    # while keeping ww/sw/noise/error from original.
    # ------------------------------------------------------------
    if args.use_comp_in_weight:
        target_weight_pad, ref_weight_pad, ref_weight_blk = compute_actual_blending_weights(
            tar_y_5d,
            compensated_refs_y_pad,  # ONLY changes diff in exponent
            ww,
            sw,
            block_size=args.bs,
        )
    else:
        target_weight_pad, ref_weight_pad, ref_weight_blk = target_weight_pad_base, ref_weight_pad_base, ref_weight_blk_base

    # Blend
    blended_no_comp_pad = blend_with_actual_weights(tar_y_5d, warped_refs_y_pad, target_weight_pad, ref_weight_pad)
    blended_comp_pad    = blend_with_actual_weights(tar_y_5d, compensated_refs_y_pad, target_weight_pad, ref_weight_pad)

    # Selection: soft gate (recommended) or fallback hard
    if args.soft_gate:
        blended_final_pad, gate_blk = soft_gate_blocks(
            target_y_pad,
            blended_no_comp_pad,
            blended_comp_pad,
            block_size=args.bs,
            tau=float(args.gate_tau),
            slope=float(args.gate_slope),
        )
        changed_blocks_soft = float((gate_blk > 0.5).float().mean().item())
    else:
        # original hard selection (kept for comparison)
        blended_final_pad, better_block_mask = select_better_compensated_blocks_hard(
            target_y_pad, blended_no_comp_pad, blended_comp_pad, block_size=args.bs
        )
        changed_blocks_soft = float(better_block_mask.float().mean().item())

    # Crop to original size
    h0, w0 = orig_hw
    target_y = target_y_pad[:, :, :h0, :w0]
    warped_base_y = warped_base_y_pad[:, :, :h0, :w0]
    blended_no_comp = blended_no_comp_pad[:, :, :h0, :w0]
    blended_comp    = blended_comp_pad[:, :, :h0, :w0]
    blended_final   = blended_final_pad[:, :, :h0, :w0]
    comp_delta = torch.abs(blended_comp - blended_no_comp)

    # Save
    save_y_png(target_y, str(out_dir / "mctf_target_y.png"))
    save_y_png(warped_base_y, str(out_dir / "mctf_warped_base_y.png"))

    save_y_png(blended_no_comp, str(out_dir / "mctf_blended_no_comp_y.png"))
    save_y_png(blended_comp,    str(out_dir / "mctf_blended_comp_y.png"))
    save_y_png(blended_final,   str(out_dir / "mctf_blended_final_y.png"))

    save_diff_png(blended_comp, blended_no_comp, str(out_dir / "mctf_diff_comp_vs_no_y.png"))
    save_diff_png(blended_final, blended_no_comp, str(out_dir / "mctf_diff_final_vs_no_y.png"))

    save_diff_png(target_y, blended_no_comp, str(out_dir / "mctf_error_no_comp_y.png"))
    save_diff_png(target_y, blended_comp,    str(out_dir / "mctf_error_comp_y.png"))
    save_diff_png(target_y, blended_final,   str(out_dir / "mctf_error_final_y.png"))

    psnr_no = compute_psnr(target_y, blended_no_comp)
    psnr_co = compute_psnr(target_y, blended_comp)
    psnr_fi = compute_psnr(target_y, blended_final)

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"PSNR no-comp  : {psnr_no:.6f} dB")
    print(f"PSNR comp     : {psnr_co:.6f} dB")
    print(f"PSNR final    : {psnr_fi:.6f} dB")
    print(f"Mean ref weight blk: {ref_weight_blk.mean().item():.6f}")
    print(f"Mean abs(comp-no): {comp_delta.mean().item():.8f}")
    print(f"Comp ref mask blk ratio (thr={args.ref_comp_weight_thr}): {ref_mask_blk.float().mean().item():.4f}")
    print(f"Changed blocks ratio (~gate>0.5): {changed_blocks_soft:.4f}")
    print(f"use_comp_in_weight={args.use_comp_in_weight}, soft_gate={args.soft_gate}, clamp_comp={not args.no_clamp_comp}")


# ============================================================
# Hard selection (kept only for fallback comparison)
# ============================================================
def select_better_compensated_blocks_hard(
    target_y: torch.Tensor,
    blended_no_comp: torch.Tensor,
    blended_with_comp: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    BS = block_size
    B, C, H, W = target_y.shape
    HB = H // BS
    WB = W // BS

    tar_blks = target_y.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    no_comp_blks = blended_no_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()
    with_comp_blks = blended_with_comp.view(B, C, HB, BS, WB, BS).permute(0, 1, 2, 4, 3, 5).contiguous()

    mse_no_comp = torch.mean((tar_blks - no_comp_blks) ** 2, dim=(1, 4, 5))
    mse_with_comp = torch.mean((tar_blks - with_comp_blks) ** 2, dim=(1, 4, 5))
    better_mask = mse_with_comp < mse_no_comp  # (B,HB,WB)

    block_mask = better_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    selected_blks = torch.where(block_mask, with_comp_blks, no_comp_blks)
    selected = selected_blks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)

    return selected, better_mask


if __name__ == "__main__":
    main()


































