#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML not installed. Install: pip install pyyaml") from e


# ============================================================
# YAML + seq cfg parsing
# ============================================================
_CFG_LINE_RE = __import__("re").compile(r"^\s*([^:#]+?)\s*:\s*(.*?)\s*$")


def load_yaml_dict(yaml_path: Path) -> Dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError("YAML root must be dict.")
    return d


def parse_seq_cfg(seq_cfg_path: Path) -> Dict[str, str]:
    if not seq_cfg_path.is_file():
        raise FileNotFoundError(seq_cfg_path)

    lines = seq_cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return {}

    out: Dict[str, str] = {}
    for raw in lines[1:]:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].rstrip()

        m = _CFG_LINE_RE.match(s)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k:
            out[k] = v
    return out


def collect_seq_items_from_yaml(
    yaml_path: Path,
    only_seq: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Expected YAML root:
      seq:
        SeqName:
          path: /path/to/original.yuv   (or yuv_path / yuv)
          seq_cls: ClassA
          seq_cfg: /path/to/seq.cfg     (optional)
          width: 3840
          height: 2160
          frames: 97
          bit_depth: 8 or 10
          fps: 60
    """
    y = load_yaml_dict(yaml_path)
    if "seq" not in y or not isinstance(y["seq"], dict):
        raise KeyError("YAML missing seq dict")

    seq_dict: Dict[str, Dict] = y["seq"]
    items: List[Dict[str, Any]] = []

    def _pick(d: Dict[str, Any], keys: List[str], default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    for seq_name, info in seq_dict.items():
        if not isinstance(info, dict):
            continue
        if only_seq is not None and seq_name not in only_seq:
            continue

        seq_cls = str(info.get("seq_cls", "NA"))
        seq_cfg = info.get("seq_cfg", None)

        width = _pick(info, ["width", "w", "source_width"], None)
        height = _pick(info, ["height", "h", "source_height"], None)
        frames = _pick(info, ["frames", "num_frames", "FrameToBeEncoded"], None)
        bit_depth = _pick(info, ["bit_depth", "bitdepth", "input_bit_depth"], None)
        fps = _pick(info, ["fps", "frame_rate", "FrameRate"], None)
        yuv_path = _pick(info, ["path", "yuv_path", "yuv"], None)

        if seq_cfg:
            cfg_path = Path(str(seq_cfg))
            if cfg_path.is_file():
                cfg = parse_seq_cfg(cfg_path)
                if width is None:
                    try:
                        width = int(cfg.get("SourceWidth", cfg.get("InputFileWidth", "")))
                    except Exception:
                        pass
                if height is None:
                    try:
                        height = int(cfg.get("SourceHeight", cfg.get("InputFileHeight", "")))
                    except Exception:
                        pass
                if frames is None:
                    try:
                        frames = int(cfg.get("FramesToBeEncoded", cfg.get("FrameToBeEncoded", "")))
                    except Exception:
                        pass
                if fps is None:
                    try:
                        fps = float(cfg.get("FrameRate", "30"))
                    except Exception:
                        pass
                if bit_depth is None:
                    try:
                        bit_depth = int(cfg.get("InputBitDepth", cfg.get("BitDepth", "10")))
                    except Exception:
                        pass
                if yuv_path is None:
                    yuv_in_cfg = cfg.get("InputFile", "").strip()
                    if yuv_in_cfg:
                        yuv_path = yuv_in_cfg

        if yuv_path is None:
            raise ValueError(f"Missing yuv path for seq={seq_name}")
        if width is None or height is None:
            raise ValueError(f"Missing width/height for seq={seq_name}")
        if frames is None:
            raise ValueError(f"Missing frames for seq={seq_name}")
        if fps is None:
            fps = 30.0
        if bit_depth is None:
            bit_depth = 10

        items.append({
            "name": str(seq_name),
            "seq_cls": str(seq_cls),
            "yuv_path": str(yuv_path),
            "width": int(width),
            "height": int(height),
            "frames": int(frames),
            "frame_rate": float(fps),
            "bit_depth": int(bit_depth),
        })

    if not items:
        raise RuntimeError("No valid sequences found in YAML after filters.")

    return items


# ============================================================
# Raw YUV IO
# ============================================================
def read_yuv420p_raw(
    path: Path,
    width: int,
    height: int,
    num_frames: int,
    bit_depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Y: [T,H,W]
      U: [T,H/2,W/2]
      V: [T,H/2,W/2]
    dtype:
      uint8  if bit_depth <= 8
      uint16 if bit_depth > 8
    """
    if bit_depth <= 8:
        sample_dtype = np.uint8
        bytes_per_sample = 1
    else:
        sample_dtype = np.dtype("<u2")
        bytes_per_sample = 2

    w2, h2 = width // 2, height // 2
    y_n = width * height
    uv_n = w2 * h2

    Y = np.empty((num_frames, height, width), dtype=sample_dtype)
    U = np.empty((num_frames, h2, w2), dtype=sample_dtype)
    V = np.empty((num_frames, h2, w2), dtype=sample_dtype)

    with open(path, "rb") as f:
        for t in range(num_frames):
            yb = f.read(y_n * bytes_per_sample)
            ub = f.read(uv_n * bytes_per_sample)
            vb = f.read(uv_n * bytes_per_sample)

            if len(yb) != y_n * bytes_per_sample or len(ub) != uv_n * bytes_per_sample or len(vb) != uv_n * bytes_per_sample:
                raise IOError(f"EOF while reading {path} at frame {t}")

            Y[t] = np.frombuffer(yb, dtype=sample_dtype).reshape(height, width)
            U[t] = np.frombuffer(ub, dtype=sample_dtype).reshape(h2, w2)
            V[t] = np.frombuffer(vb, dtype=sample_dtype).reshape(h2, w2)

    return Y, U, V


# ============================================================
# Numeric helpers
# ============================================================
def to_float01(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    maxv = float((1 << bit_depth) - 1)
    arr = arr.astype(np.float32) / maxv
    return np.clip(arr, 0.0, 1.0)


def mse01(a01: np.ndarray, b01: np.ndarray) -> float:
    d = a01.astype(np.float32) - b01.astype(np.float32)
    return float(np.mean(d * d))


def psnr_from_mse(m: float, eps: float = 1e-12) -> float:
    if m < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / m)


def psnr_per_frame_mean(a01: np.ndarray, b01: np.ndarray) -> float:
    """
    a01, b01: [T,H,W]
    frame-wise PSNR average
    """
    assert a01.shape == b01.shape and a01.ndim == 3
    T = a01.shape[0]
    s = 0.0
    for t in range(T):
        s += psnr_from_mse(mse01(a01[t], b01[t]))
    return s / max(T, 1)


def bin_to_kbps(bin_path: Path, frames: int, fps: float) -> float:
    bits = os.path.getsize(bin_path) * 8.0
    bps = bits * (fps / frames)
    return bps / 1000.0


# ============================================================
# Path helpers
# ============================================================
def find_qp_dirs(codec_out: Path) -> List[Path]:
    return sorted([p for p in codec_out.iterdir() if p.is_dir() and p.name.startswith("qp")])


def parse_qp_from_dirname(name: str) -> int:
    if not name.startswith("qp"):
        raise ValueError(f"Bad qp dir name: {name}")
    return int(name[2:])


def bin_path_for_seq(codec_out: Path, qp: int, seq_cls: str, seq_name: str) -> Path:
    return codec_out / f"qp{qp:02d}" / "bin" / seq_cls / f"{seq_name}.bin"


def rec_path_for_seq(codec_out: Path, qp: int, seq_cls: str, seq_name: str) -> Path:
    return codec_out / f"qp{qp:02d}" / "rec" / seq_cls / f"{seq_name}.yuv"


def rec_post_path_for_seq(codec_out: Path, qp: int, seq_cls: str, seq_name: str) -> Path:
    return codec_out / f"qp{qp:02d}" / "rec_post" / seq_cls / f"{seq_name}.yuv"


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--yaml", type=str, required=True, help="dataset yaml (seq dict)")
    ap.add_argument("--codec_out", type=str, required=True, help="codec output root containing qpXX/bin, rec, rec_post")
    ap.add_argument("--out_csv", type=str, required=True, help="output csv path")

    ap.add_argument("--only_seq", type=str, default="", help="comma-separated seq names")
    ap.add_argument("--frame_limit", type=int, default=0, help="0=use all frames from yaml, >0=use first N frames")
    ap.add_argument("--eval_rec", action="store_true", help="also evaluate rec against original")
    ap.add_argument("--eval_rec_post", action="store_true", help="also evaluate rec_post against original")

    args = ap.parse_args()

    if not args.eval_rec and not args.eval_rec_post:
        args.eval_rec = True
        args.eval_rec_post = True

    codec_out = Path(args.codec_out)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    only_seq = {s.strip() for s in args.only_seq.split(",") if s.strip()}
    seq_items = collect_seq_items_from_yaml(
        Path(args.yaml),
        only_seq=only_seq if only_seq else None,
    )

    qp_dirs = find_qp_dirs(codec_out)
    if not qp_dirs:
        raise RuntimeError(f"No qpXX dirs found under: {codec_out}")

    rows: List[str] = []
    rows.append(
        "qp,seq_cls,seq_name,frames,fps,kbps,"
        "psnrY_rec,psnrU_rec,psnrV_rec,"
        "psnrY_rec_post,psnrU_rec_post,psnrV_rec_post"
    )

    total = len(qp_dirs) * len(seq_items)
    done = 0

    for qp_dir in qp_dirs:
        qp = parse_qp_from_dirname(qp_dir.name)

        for item in seq_items:
            done += 1

            seq_name = item["name"]
            seq_cls = item["seq_cls"]
            width = int(item["width"])
            height = int(item["height"])
            total_frames = int(item["frames"])
            fps = float(item["frame_rate"])
            bit_depth = int(item["bit_depth"])
            org_path = Path(item["yuv_path"])

            used_frames = total_frames if args.frame_limit <= 0 else min(args.frame_limit, total_frames)

            bin_path = bin_path_for_seq(codec_out, qp, seq_cls, seq_name)
            rec_path = rec_path_for_seq(codec_out, qp, seq_cls, seq_name)
            rec_post_path = rec_post_path_for_seq(codec_out, qp, seq_cls, seq_name)

            print(f"[{done}/{total}] qp={qp} seq={seq_name}")

            if not org_path.is_file():
                print(f"  [SKIP] missing original: {org_path}")
                continue
            if not bin_path.is_file():
                print(f"  [SKIP] missing bin: {bin_path}")
                continue

            # original
            oY_raw, oU_raw, oV_raw = read_yuv420p_raw(
                path=org_path,
                width=width,
                height=height,
                num_frames=used_frames,
                bit_depth=bit_depth,
            )
            oY = to_float01(oY_raw, bit_depth)
            oU = to_float01(oU_raw, bit_depth)
            oV = to_float01(oV_raw, bit_depth)

            kbps = bin_to_kbps(bin_path, frames=used_frames, fps=fps)

            psnrY_rec = float("nan")
            psnrU_rec = float("nan")
            psnrV_rec = float("nan")

            psnrY_rec_post = float("nan")
            psnrU_rec_post = float("nan")
            psnrV_rec_post = float("nan")

            if args.eval_rec:
                if rec_path.is_file():
                    rY_raw, rU_raw, rV_raw = read_yuv420p_raw(
                        path=rec_path,
                        width=width,
                        height=height,
                        num_frames=used_frames,
                        bit_depth=10,  # codec rec assumed 10-bit
                    )
                    rY = to_float01(rY_raw, 10)
                    rU = to_float01(rU_raw, 10)
                    rV = to_float01(rV_raw, 10)

                    psnrY_rec = psnr_per_frame_mean(rY, oY)
                    psnrU_rec = psnr_per_frame_mean(rU, oU)
                    psnrV_rec = psnr_per_frame_mean(rV, oV)
                else:
                    print(f"  [WARN] missing rec: {rec_path}")

            if args.eval_rec_post:
                if rec_post_path.is_file():
                    pY_raw, pU_raw, pV_raw = read_yuv420p_raw(
                        path=rec_post_path,
                        width=width,
                        height=height,
                        num_frames=used_frames,
                        bit_depth=10,  # rec_post also 10-bit
                    )
                    pY = to_float01(pY_raw, 10)
                    pU = to_float01(pU_raw, 10)
                    pV = to_float01(pV_raw, 10)

                    psnrY_rec_post = psnr_per_frame_mean(pY, oY)
                    psnrU_rec_post = psnr_per_frame_mean(pU, oU)
                    psnrV_rec_post = psnr_per_frame_mean(pV, oV)
                else:
                    print(f"  [WARN] missing rec_post: {rec_post_path}")

            rows.append(
                f"{qp},{seq_cls},{seq_name},{used_frames},{fps:.6f},{kbps:.10f},"
                f"{psnrY_rec:.6f},{psnrU_rec:.6f},{psnrV_rec:.6f},"
                f"{psnrY_rec_post:.6f},{psnrU_rec_post:.6f},{psnrV_rec_post:.6f}"
            )

    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Done. CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
