import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch


# =========================================================
# 1. Original MCTF param function (almost 그대로 사용)
# =========================================================
def calculate_mctf_params(tar_y, ref_y, BS=16):
    """
    tar_y: (B, 1, 1, H, W) - 현재 프레임
    ref_y: (B, 8, 1, H, W) - 참조 프레임들
    BS: 블록 사이즈 (현재 H=W=BS 라고 가정)
    """
    B, Ref_num, C, H, W = ref_y.shape

    # 1. 10-bit scaling (0~1 -> 0~1023)
    tar = tar_y * 1023.0
    ref = ref_y * 1023.0

    # 2. block view
    tar_blks = tar.view(B, 1, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5)
    ref_blks = ref.view(B, Ref_num, H // BS, BS, W // BS, BS).permute(0, 1, 2, 4, 3, 5)

    # 3. constants
    offset = 5.0
    scale = 50.0
    cntV = BS * BS
    cntD = 2 * cntV - BS - BS

    tar_avg = torch.mean(tar_blks, dim=(4, 5), keepdim=True)
    tar_var = torch.sum((tar_blks - tar_avg) ** 2, dim=(4, 5))

    # 4. error / noise
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


# =========================================================
# 2. YUV420p10le reader (Y only)
# =========================================================
def read_yuv420p10le_y_only(path: str, width: int, height: int, num_frames: int) -> np.ndarray:
    """
    Returns:
        Y: (T, H, W), uint16, range 0~1023 expected
    """
    path = str(path)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_samples = y_size + uv_size + uv_size  # 4:2:0
    frame_bytes = frame_samples * 2  # 10-bit stored in 16-bit little-endian

    expected_bytes = frame_bytes * num_frames
    actual_bytes = os.path.getsize(path)
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"File too small: {path}\n"
            f"Expected at least {expected_bytes} bytes for {num_frames} frames, got {actual_bytes}"
        )

    raw = np.fromfile(path, dtype='<u2', count=frame_samples * num_frames)
    raw = raw.reshape(num_frames, frame_samples)

    y = raw[:, :y_size].reshape(num_frames, height, width)
    return y


# =========================================================
# 3. Reference frame selection
# =========================================================
def build_ref_indices(t: int, num_frames: int, temporal_offsets=(-4, -3, -2, -1, 1, 2, 3, 4)) -> List[int]:
    """
    현재 frame t에 대해 8개 ref index를 만든다.
    경계는 clamp.
    """
    refs = []
    for off in temporal_offsets:
        idx = t + off
        idx = max(0, min(num_frames - 1, idx))
        refs.append(idx)
    return refs


# =========================================================
# 4. Block noise extractor
# =========================================================
def compute_block_noise_and_error(
    clip_y: np.ndarray,
    t: int,
    x: int,
    y: int,
    bs: int,
    device: str = "cpu",
) -> Tuple[float, float, int]:
    """
    clip_y: (T,H,W), uint16
    t: target frame index
    x,y: left-top
    bs: block size

    Returns:
        selected_noise: best ref(최소 error ref)의 noise
        selected_error: min error
        selected_ref_idx_in_8: 0~7
    """
    T, H, W = clip_y.shape
    assert x + bs <= W and y + bs <= H

    tar_blk = clip_y[t, y:y+bs, x:x+bs].astype(np.float32) / 1023.0
    ref_indices = build_ref_indices(t, T)

    ref_blks = []
    for ridx in ref_indices:
        ref_blk = clip_y[ridx, y:y+bs, x:x+bs].astype(np.float32) / 1023.0
        ref_blks.append(ref_blk)

    tar_tensor = torch.from_numpy(tar_blk).to(device).view(1, 1, 1, bs, bs)
    ref_tensor = torch.from_numpy(np.stack(ref_blks, axis=0)).to(device).view(1, 8, 1, bs, bs)

    with torch.no_grad():
        noise, error, _, _ = calculate_mctf_params(tar_tensor, ref_tensor, BS=bs)

    # noise/error shape: (B,8,1,1) or (B,8,H/BS,W/BS) == (1,8,1,1)
    noise_1d = noise.view(8).cpu().numpy()
    error_1d = error.view(8).cpu().numpy()

    best_idx = int(np.argmin(error_1d))
    selected_noise = float(noise_1d[best_idx])
    selected_error = float(error_1d[best_idx])

    return selected_noise, selected_error, best_idx


# =========================================================
# 5. Quadtree node
# =========================================================
@dataclass
class QuadNode:
    frame_idx: int
    x: int
    y: int
    bs: int
    noise: float
    error: float
    best_ref_idx: int
    cost: float
    split: bool
    children: List[Any]


# =========================================================
# 6. Quadtree optimization
# =========================================================
def optimize_quadtree_block(
    clip_y: np.ndarray,
    frame_idx: int,
    x: int,
    y: int,
    bs: int,
    target_noise: float,
    min_bs: int,
    split_penalty: float,
    device: str = "cpu",
    memo: Dict[Tuple[int, int, int, int], Tuple[float, float, int]] = None,
) -> QuadNode:
    """
    하나의 block에 대해
    - keep할지
    - 4-way split할지
    최적화한다.
    """
    if memo is None:
        memo = {}

    key = (frame_idx, x, y, bs)
    if key in memo:
        noise_val, error_val, best_ref_idx = memo[key]
    else:
        noise_val, error_val, best_ref_idx = compute_block_noise_and_error(
            clip_y=clip_y,
            t=frame_idx,
            x=x,
            y=y,
            bs=bs,
            device=device,
        )
        memo[key] = (noise_val, error_val, best_ref_idx)

    keep_cost = abs(noise_val - target_noise)

    # 더 이상 못 쪼개면 leaf
    if bs <= min_bs:
        return QuadNode(
            frame_idx=frame_idx,
            x=x,
            y=y,
            bs=bs,
            noise=noise_val,
            error=error_val,
            best_ref_idx=best_ref_idx,
            cost=keep_cost,
            split=False,
            children=[],
        )

    half = bs // 2
    child_coords = [
        (x, y),
        (x + half, y),
        (x, y + half),
        (x + half, y + half),
    ]

    children = [
        optimize_quadtree_block(
            clip_y=clip_y,
            frame_idx=frame_idx,
            x=cx,
            y=cy,
            bs=half,
            target_noise=target_noise,
            min_bs=min_bs,
            split_penalty=split_penalty,
            device=device,
            memo=memo,
        )
        for cx, cy in child_coords
    ]

    split_cost = sum(ch.cost for ch in children) + split_penalty

    if keep_cost <= split_cost:
        return QuadNode(
            frame_idx=frame_idx,
            x=x,
            y=y,
            bs=bs,
            noise=noise_val,
            error=error_val,
            best_ref_idx=best_ref_idx,
            cost=keep_cost,
            split=False,
            children=[],
        )
    else:
        return QuadNode(
            frame_idx=frame_idx,
            x=x,
            y=y,
            bs=bs,
            noise=noise_val,
            error=error_val,
            best_ref_idx=best_ref_idx,
            cost=split_cost,
            split=True,
            children=children,
        )


def collect_leaf_nodes(node: QuadNode) -> List[QuadNode]:
    if not node.split:
        return [node]
    out = []
    for ch in node.children:
        out.extend(collect_leaf_nodes(ch))
    return out


# =========================================================
# 7. Full-frame partition
# =========================================================
def optimize_frame_partition(
    clip_y: np.ndarray,
    frame_idx: int,
    target_noise: float,
    root_bs: int = 64,
    min_bs: int = 8,
    split_penalty: float = 1.0,
    width: int = 256,
    height: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    프레임 전체를 root_bs 타일 단위로 나누고,
    각 타일에서 quadtree 최적화 수행.
    """
    assert width % root_bs == 0 and height % root_bs == 0
    assert root_bs % min_bs == 0

    roots: List[QuadNode] = []
    memo = {}

    for y in range(0, height, root_bs):
        for x in range(0, width, root_bs):
            root = optimize_quadtree_block(
                clip_y=clip_y,
                frame_idx=frame_idx,
                x=x,
                y=y,
                bs=root_bs,
                target_noise=target_noise,
                min_bs=min_bs,
                split_penalty=split_penalty,
                device=device,
                memo=memo,
            )
            roots.append(root)

    leaves = []
    for r in roots:
        leaves.extend(collect_leaf_nodes(r))

    avg_abs_diff = float(np.mean([abs(n.noise - target_noise) for n in leaves])) if leaves else 0.0

    result = {
        "frame_idx": frame_idx,
        "target_noise": target_noise,
        "root_bs": root_bs,
        "min_bs": min_bs,
        "split_penalty": split_penalty,
        "num_leaf_blocks": len(leaves),
        "mean_abs_noise_diff": avg_abs_diff,
        "leaf_blocks": [
            {
                "frame_idx": n.frame_idx,
                "x": n.x,
                "y": n.y,
                "bs": n.bs,
                "noise": n.noise,
                "error": n.error,
                "best_ref_idx": n.best_ref_idx,
                "abs_diff_to_target": abs(n.noise - target_noise),
            }
            for n in leaves
        ],
    }
    return result


# =========================================================
# 8. Visualization helper (optional)
# =========================================================
def render_partition_map(frame_result: Dict[str, Any], width: int = 256, height: int = 256) -> np.ndarray:
    """
    각 leaf block size를 map으로 저장.
    픽셀 위치마다 그 block의 bs 값이 들어간다.
    """
    out = np.zeros((height, width), dtype=np.int32)
    for blk in frame_result["leaf_blocks"]:
        x, y, bs = blk["x"], blk["y"], blk["bs"]
        out[y:y+bs, x:x+bs] = bs
    return out


# =========================================================
# 9. Save helpers
# =========================================================
def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_partition_map_npy(frame_result: Dict[str, Any], out_path: Path, width: int = 256, height: int = 256):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_map = render_partition_map(frame_result, width=width, height=height)
    np.save(out_path, part_map)


# =========================================================
# 10. Main processing for one clip
# =========================================================
def process_clip(
    clip_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    num_frames: int,
    target_noise: float,
    root_bs: int,
    min_bs: int,
    split_penalty: float,
    frame_start: int,
    frame_end: int,
    device: str,
):
    print(f"[INFO] Reading clip: {clip_path}")
    clip_y = read_yuv420p10le_y_only(clip_path, width=width, height=height, num_frames=num_frames)

    clip_name = clip_path.stem
    clip_out_dir = out_dir / clip_name
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "clip_name": clip_name,
        "clip_path": str(clip_path),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "target_noise": target_noise,
        "root_bs": root_bs,
        "min_bs": min_bs,
        "split_penalty": split_penalty,
        "frames": [],
    }

    for t in range(frame_start, frame_end + 1):
        print(f"  [INFO] Frame {t}")
        frame_result = optimize_frame_partition(
            clip_y=clip_y,
            frame_idx=t,
            target_noise=target_noise,
            root_bs=root_bs,
            min_bs=min_bs,
            split_penalty=split_penalty,
            width=width,
            height=height,
            device=device,
        )

        summary["frames"].append({
            "frame_idx": frame_result["frame_idx"],
            "num_leaf_blocks": frame_result["num_leaf_blocks"],
            "mean_abs_noise_diff": frame_result["mean_abs_noise_diff"],
        })

        save_json(frame_result, clip_out_dir / f"frame_{t:03d}.json")
        save_partition_map_npy(frame_result, clip_out_dir / f"frame_{t:03d}_partition.npy", width=width, height=height)

    save_json(summary, clip_out_dir / "summary.json")


# =========================================================
# 11. CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing *.yuv")
    p.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=33)

    p.add_argument("--target_noise", type=float, required=True, help="Target constant noise value")
    p.add_argument("--root_bs", type=int, default=64, help="Initial root block size")
    p.add_argument("--min_bs", type=int, default=8, help="Minimum block size")
    p.add_argument("--split_penalty", type=float, default=1.0, help="Penalty for splitting a block")

    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_end", type=int, default=32)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert args.width == 256 and args.height == 256, "현재 요청 기준으로 256x256에 맞춰 작성함."
    assert args.root_bs in [64, 32, 16, 8]
    assert args.min_bs in [64, 32, 16, 8]
    assert args.root_bs >= args.min_bs
    assert args.frame_start >= 0 and args.frame_end < args.num_frames

    yuv_list = sorted(input_dir.glob("*.yuv"))
    if not yuv_list:
        raise FileNotFoundError(f"No .yuv files found in {input_dir}")

    for clip_path in yuv_list:
        process_clip(
            clip_path=clip_path,
            out_dir=output_dir,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            target_noise=args.target_noise,
            root_bs=args.root_bs,
            min_bs=args.min_bs,
            split_penalty=args.split_penalty,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            device=args.device,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()












def draw_partition_on_y(frame_y: np.ndarray, leaf_blocks: List[Dict[str, Any]], line_value: int = 1023, thickness: int = 1) -> np.ndarray:
    """
    frame_y: (H, W), uint16
    leaf_blocks: frame_result["leaf_blocks"]
    line_value: 경계선 밝기 (10-bit: 0~1023)
    thickness: 선 두께
    """
    out = frame_y.copy()
    H, W = out.shape

    for blk in leaf_blocks:
        x = int(blk["x"])
        y = int(blk["y"])
        bs = int(blk["bs"])

        x0, y0 = x, y
        x1, y1 = x + bs - 1, y + bs - 1

        # top
        yy0 = max(0, y0)
        yy1 = min(H, y0 + thickness)
        xx0 = max(0, x0)
        xx1 = min(W, x1 + 1)
        out[yy0:yy1, xx0:xx1] = line_value

        # bottom
        yy0 = max(0, y1 - thickness + 1)
        yy1 = min(H, y1 + 1)
        xx0 = max(0, x0)
        xx1 = min(W, x1 + 1)
        out[yy0:yy1, xx0:xx1] = line_value

        # left
        yy0 = max(0, y0)
        yy1 = min(H, y1 + 1)
        xx0 = max(0, x0)
        xx1 = min(W, x0 + thickness)
        out[yy0:yy1, xx0:xx1] = line_value

        # right
        yy0 = max(0, y0)
        yy1 = min(H, y1 + 1)
        xx0 = max(0, x1 - thickness + 1)
        xx1 = min(W, x1 + 1)
        out[yy0:yy1, xx0:xx1] = line_value

    return out


def save_yuv420p10le_from_y_frames(y_frames: np.ndarray, out_path: Path):
    """
    y_frames: (T,H,W), uint16
    U/V는 neutral chroma(512)로 채워 저장
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T, H, W = y_frames.shape
    uv_h, uv_w = H // 2, W // 2
    u_plane = np.full((uv_h, uv_w), 512, dtype=np.uint16)
    v_plane = np.full((uv_h, uv_w), 512, dtype=np.uint16)

    with open(out_path, "wb") as f:
        for t in range(T):
            y = np.clip(y_frames[t], 0, 1023).astype(np.uint16)
            y.astype("<u2").tofile(f)
            u_plane.astype("<u2").tofile(f)
            v_plane.astype("<u2").tofile(f)






def process_clip(
    clip_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    num_frames: int,
    target_noise: float,
    root_bs: int,
    min_bs: int,
    split_penalty: float,
    frame_start: int,
    frame_end: int,
    device: str,
):
    print(f"[INFO] Reading clip: {clip_path}")
    clip_y = read_yuv420p10le_y_only(clip_path, width=width, height=height, num_frames=num_frames)

    clip_name = clip_path.stem
    clip_out_dir = out_dir / clip_name
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "clip_name": clip_name,
        "clip_path": str(clip_path),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "target_noise": target_noise,
        "root_bs": root_bs,
        "min_bs": min_bs,
        "split_penalty": split_penalty,
        "frames": [],
    }

    # 전체 33프레임짜리 overlay clip 저장용
    vis_y_frames = clip_y.copy()

    for t in range(frame_start, frame_end + 1):
        print(f"  [INFO] Frame {t}")
        frame_result = optimize_frame_partition(
            clip_y=clip_y,
            frame_idx=t,
            target_noise=target_noise,
            root_bs=root_bs,
            min_bs=min_bs,
            split_penalty=split_penalty,
            width=width,
            height=height,
            device=device,
        )

        summary["frames"].append({
            "frame_idx": frame_result["frame_idx"],
            "num_leaf_blocks": frame_result["num_leaf_blocks"],
            "mean_abs_noise_diff": frame_result["mean_abs_noise_diff"],
        })

        save_json(frame_result, clip_out_dir / f"frame_{t:03d}.json")

        # partition line을 원본 Y 위에 그림
        vis_y_frames[t] = draw_partition_on_y(
            frame_y=clip_y[t],
            leaf_blocks=frame_result["leaf_blocks"],
            line_value=1023,   # 흰색 선
            thickness=1,
        )

    # 33프레임짜리 시각화 yuv 저장
    save_yuv420p10le_from_y_frames(
        vis_y_frames,
        clip_out_dir / f"{clip_name}_partition_overlay.yuv"
    )

    save_json(summary, clip_out_dir / "summary.json")






