#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

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
          path: /path/to/fullres.yuv      (or yuv_path / yuv)
          seq_cls: ClassA
          seq_cfg: /path/to/seq.cfg       (optional)
          width: 3840                     (optional if seq_cfg exists)
          height: 2160                    (optional if seq_cfg exists)
          frames: 97                      (optional if seq_cfg exists)
          bit_depth: 8 or 10              (optional if seq_cfg exists)
          fps: 60                         (optional if seq_cfg exists)
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
# YUV420 raw IO
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
      uint8 if bit_depth <= 8
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


def save_yuv420p10le(
    out_path: Path,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
):
    """
    Save planar yuv420p10le.
    Y: [T,H,W] uint16
    U: [T,H/2,W/2] uint16
    V: [T,H/2,W/2] uint16
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if Y.dtype != np.uint16 or U.dtype != np.uint16 or V.dtype != np.uint16:
        raise TypeError("Y/U/V must be uint16")

    with open(out_path, "wb") as f:
        for t in range(Y.shape[0]):
            Y[t].astype("<u2", copy=False).tofile(f)
            U[t].astype("<u2", copy=False).tofile(f)
            V[t].astype("<u2", copy=False).tofile(f)


# ============================================================
# Numeric conversion
# ============================================================
def to_float01(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    maxv = float((1 << bit_depth) - 1)
    arr = arr.astype(np.float32) / maxv
    return np.clip(arr, 0.0, 1.0)


def float01_to_uint10(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    return np.round(arr * 1023.0).astype(np.uint16)


# ============================================================
# Model
# ============================================================
def build_model():
    """
    Replace this with your actual blur network loader.

    Expected:
        y_out, u_out, v_out = model(y, u, v)

    Inputs:
        y: [B,3,1,H,W]
        u: [B,3,1,H/2,W/2]
        v: [B,3,1,H/2,W/2]

    Outputs:
        same shapes
    """
    raise NotImplementedError("Replace build_model() with your actual model loader.")


# ============================================================
# Temporal reflect indexing
# ============================================================
def reflect_index_1d(i: int, n: int) -> int:
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


def get_triplet_indices_no_overlap(start: int, T: int) -> List[int]:
    """
    Non-overlap 3-frame chunk starting at start.
    Missing frames are reflect padded.
    """
    return [
        reflect_index_1d(start + 0, T),
        reflect_index_1d(start + 1, T),
        reflect_index_1d(start + 2, T),
    ]


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
def run_model_on_triplet(
    model: torch.nn.Module,
    Y3: np.ndarray,
    U3: np.ndarray,
    V3: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
      Y3: [3,H,W]
      U3: [3,H/2,W/2]
      V3: [3,H/2,W/2]

    Returns:
      Y_out: [3,H,W]
      U_out: [3,H/2,W/2]
      V_out: [3,H/2,W/2]
    """
    y = torch.from_numpy(Y3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
    u = torch.from_numpy(U3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H/2,W/2]
    v = torch.from_numpy(V3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H/2,W/2]

    y, (py, px) = pad_btchw_to_multiple_of_16(y)
    u, (puy, pux) = pad_btchw_to_multiple_of_16(u)
    v, (pvy, pvx) = pad_btchw_to_multiple_of_16(v)

    y_out, u_out, v_out = model(y, u, v)

    y_out = crop_btchw(y_out, py, px)
    u_out = crop_btchw(u_out, puy, pux)
    v_out = crop_btchw(v_out, pvy, pvx)

    y_out = y_out.squeeze(0).squeeze(1).detach().cpu().numpy()
    u_out = u_out.squeeze(0).squeeze(1).detach().cpu().numpy()
    v_out = v_out.squeeze(0).squeeze(1).detach().cpu().numpy()

    return y_out, u_out, v_out


@torch.no_grad()
def process_one_sequence_nonoverlap(
    model: torch.nn.Module,
    yuv_path: Path,
    width: int,
    height: int,
    used_frames: int,
    bit_depth: int,
    out_yuv_path: Path,
    device: torch.device,
):
    """
    Non-overlap 3-frame processing.

    Example:
      start=0 -> frames [0,1,2]
      start=3 -> frames [3,4,5]
      ...
      last chunk 부족하면 reflect padding

    Output length = used_frames
    Output file format = always yuv420p10le
    """
    Y_raw, U_raw, V_raw = read_yuv420p_raw(
        path=yuv_path,
        width=width,
        height=height,
        num_frames=used_frames,
        bit_depth=bit_depth,
    )

    Y = to_float01(Y_raw, bit_depth)
    U = to_float01(U_raw, bit_depth)
    V = to_float01(V_raw, bit_depth)

    T = used_frames
    H, W = Y.shape[1], Y.shape[2]
    H2, W2 = U.shape[1], U.shape[2]

    outY = np.empty((T, H, W), dtype=np.float32)
    outU = np.empty((T, H2, W2), dtype=np.float32)
    outV = np.empty((T, H2, W2), dtype=np.float32)

    for start in range(0, T, 3):
        idxs = get_triplet_indices_no_overlap(start, T)

        Y3 = Y[idxs]
        U3 = U[idxs]
        V3 = V[idxs]

        Yo, Uo, Vo = run_model_on_triplet(
            model=model,
            Y3=Y3,
            U3=U3,
            V3=V3,
            device=device,
        )

        valid_len = min(3, T - start)
        outY[start:start + valid_len] = Yo[:valid_len]
        outU[start:start + valid_len] = Uo[:valid_len]
        outV[start:start + valid_len] = Vo[:valid_len]

    save_yuv420p10le(
        out_yuv_path,
        float01_to_uint10(outY),
        float01_to_uint10(outU),
        float01_to_uint10(outV),
    )


# ============================================================
# Encoder helpers
# ============================================================
def parse_qp_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("Empty --qps")
    return vals


def build_encoder_cmd(
    encoder_app: Path,
    cfg_path: Path,
    input_yuv: Path,
    bitstream_path: Path,
    recon_path: Path,
    log_path: Path,
    width: int,
    height: int,
    frames: int,
    qp: int,
    intra_period: int,
    frame_rate: int,
    input_bit_depth: int = 10,
) -> str:
    cmd = (
        f"{shlex.quote(str(encoder_app))} "
        f"-c {shlex.quote(str(cfg_path))} "
        f"-i {shlex.quote(str(input_yuv))} "
        f"-b {shlex.quote(str(bitstream_path))} "
        f"-o {shlex.quote(str(recon_path))} "
        f"-wdt {width} "
        f"-hgt {height} "
        f"-fr {frame_rate} "
        f"-f {frames} "
        f"-q {qp} "
        f"--InputBitDepth={input_bit_depth} "
        f"--IntraPeriod={intra_period} "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )
    return cmd


def submit_bsub_batch(
    cmds: List[str],
    job_name: str,
    queue: str = "",
    extra_bsub_args: str = "",
    dry_run: bool = False,
):
    if not cmds:
        return

    body = " ".join([f"({cmd})&" for cmd in cmds]) + " wait"

    parts = ["bsub", "-J", job_name]
    if queue:
        parts += ["-q", queue]
    if extra_bsub_args.strip():
        parts += shlex.split(extra_bsub_args)
    parts.append(body)

    print("[BSUB CMD]")
    print(" ".join(shlex.quote(p) for p in parts))

    if dry_run:
        return

    subprocess.run(parts, check=True)


def build_codec_paths(
    blur_yuv_path: Path,
    blur_yuv_root: Path,
    codec_root: Path,
    qp: int,
) -> Tuple[Path, Path, Path]:
    """
    codec_root/
      qp22/
        bin/<seq_cls>/<seq_name>.bin
        rec/<seq_cls>/<seq_name>.yuv
        log/<seq_cls>/<seq_name>.log
    """
    rel_no_suffix = blur_yuv_path.relative_to(blur_yuv_root).with_suffix("")

    bit_path = codec_root / f"qp{qp:02d}" / "bin" / rel_no_suffix.with_suffix(".bin")
    rec_path = codec_root / f"qp{qp:02d}" / "rec" / rel_no_suffix.with_suffix(".yuv")
    log_path = codec_root / f"qp{qp:02d}" / "log" / rel_no_suffix.with_suffix(".log")

    bit_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return bit_path, rec_path, log_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # metadata
    parser.add_argument("--yaml", type=str, required=True, help="dataset yaml (seq dict)")

    # blur inference
    parser.add_argument("--blur_out_root", type=str, required=True, help="Root folder to save blurred full-res yuvs")
    parser.add_argument("--tag", type=str, required=True, help="Experiment tag")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # frame control
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=0,
        help="0: use all frames from yaml. >0: use first N frames for every sequence."
    )

    # filters
    parser.add_argument("--only_seq", type=str, default="", help="comma-separated seq names")

    # encoder submit
    parser.add_argument("--submit_bsub", action="store_true")
    parser.add_argument("--cfg_path", type=str, default="")
    parser.add_argument("--bin_dir", type=str, default="")
    parser.add_argument("--encoder_name", type=str, default="EncoderApp")
    parser.add_argument("--codec_root", type=str, default="", help="Root for codec bit/rec/log outputs")

    parser.add_argument("--qps", type=str, required=True, help='Comma-separated QPs, e.g. "22,27,32,37"')
    parser.add_argument("--batch_size", type=int, default=8, help="How many encoder commands per bsub submission")
    parser.add_argument("--job_prefix", type=str, default="HEVC_8CPU")
    parser.add_argument("--queue", type=str, default="")
    parser.add_argument("--extra_bsub_args", type=str, default="", help='e.g. \'-n 8 -R "span[hosts=1]"\'')
    parser.add_argument("--dry_run_bsub", action="store_true")

    # encoder params
    parser.add_argument("--intra_period", type=int, default=32)

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    qps = parse_qp_list(args.qps)

    only_seq = {s.strip() for s in args.only_seq.split(",") if s.strip()}
    seq_items = collect_seq_items_from_yaml(
        Path(args.yaml),
        only_seq=only_seq if only_seq else None,
    )

    blur_root = Path(args.blur_out_root) / args.tag
    blur_yuv_root = blur_root / "blur_yuv"
    blur_yuv_root.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model = model.to(device)
    model.eval()

    encoder_app = None
    cfg_path = None
    codec_root = None

    if args.submit_bsub:
        if not args.cfg_path or not args.bin_dir or not args.codec_root:
            raise ValueError(
                "When --submit_bsub is used, you must provide "
                "--cfg_path, --bin_dir, --codec_root"
            )

        cfg_path = Path(args.cfg_path)
        encoder_app = Path(args.bin_dir) / args.encoder_name
        codec_root = Path(args.codec_root) / args.tag

        if not cfg_path.exists():
            raise FileNotFoundError(f"cfg_path not found: {cfg_path}")
        if not encoder_app.exists():
            raise FileNotFoundError(f"EncoderApp not found: {encoder_app}")

    pending_cmds: List[str] = []
    submitted_job_count = 0

    for i, item in enumerate(seq_items, 1):
        seq_name = item["name"]
        seq_cls = item["seq_cls"]
        yuv_path = Path(item["yuv_path"])
        width = int(item["width"])
        height = int(item["height"])
        total_frames = int(item["frames"])
        fps = float(item["frame_rate"])
        bit_depth = int(item["bit_depth"])

        used_frames = total_frames if args.frame_limit <= 0 else min(args.frame_limit, total_frames)
        if used_frames <= 0:
            print(f"[SKIP] {seq_name}: used_frames <= 0")
            continue

        if not yuv_path.exists():
            print(f"[SKIP] missing yuv: {yuv_path}")
            continue

        out_yuv_path = blur_yuv_root / seq_cls / f"{seq_name}.yuv"

        print(
            f"[{i}/{len(seq_items)}] Blur infer: {seq_name} | "
            f"frames={used_frames}/{total_frames} | "
            f"bit_depth={bit_depth} | path={yuv_path}"
        )

        process_one_sequence_nonoverlap(
            model=model,
            yuv_path=yuv_path,
            width=width,
            height=height,
            used_frames=used_frames,
            bit_depth=bit_depth,
            out_yuv_path=out_yuv_path,
            device=device,
        )

        if args.submit_bsub:
            for qp in qps:
                bit_path, rec_path, log_path = build_codec_paths(
                    blur_yuv_path=out_yuv_path,
                    blur_yuv_root=blur_yuv_root,
                    codec_root=codec_root,
                    qp=qp,
                )

                cmd = build_encoder_cmd(
                    encoder_app=encoder_app,
                    cfg_path=cfg_path,
                    input_yuv=out_yuv_path,
                    bitstream_path=bit_path,
                    recon_path=rec_path,
                    log_path=log_path,
                    width=width,
                    height=height,
                    frames=used_frames,
                    qp=qp,
                    intra_period=args.intra_period,
                    frame_rate=int(round(fps)),
                    input_bit_depth=10,  # blurred output is always 10-bit
                )
                pending_cmds.append(cmd)

                if len(pending_cmds) >= args.batch_size:
                    job_name = f"{args.job_prefix}_{submitted_job_count:04d}"
                    submit_bsub_batch(
                        cmds=pending_cmds,
                        job_name=job_name,
                        queue=args.queue,
                        extra_bsub_args=args.extra_bsub_args,
                        dry_run=args.dry_run_bsub,
                    )
                    submitted_job_count += 1
                    pending_cmds = []

    if args.submit_bsub and pending_cmds:
        job_name = f"{args.job_prefix}_{submitted_job_count:04d}"
        submit_bsub_batch(
            cmds=pending_cmds,
            job_name=job_name,
            queue=args.queue,
            extra_bsub_args=args.extra_bsub_args,
            dry_run=args.dry_run_bsub,
        )
        submitted_job_count += 1

    print(f"[INFO] Blurred YUVs saved under: {blur_yuv_root}")
    if args.submit_bsub:
        print(f"[INFO] Submitted {submitted_job_count} bsub jobs")
        print(f"[INFO] Codec outputs root: {codec_root}")


if __name__ == "__main__":
    main()
