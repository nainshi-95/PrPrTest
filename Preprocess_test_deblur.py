#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
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
          seq_cls: ClassA
          seq_cfg: /path/to/seq.cfg   (optional)
          width: 3840
          height: 2160
          frames: 97
          bit_depth: 10
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
# Raw YUV420p10le IO
# ============================================================
def read_yuv420p10le(
    path: Path,
    width: int,
    height: int,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns uint16:
      Y: [T,H,W]
      U: [T,H/2,W/2]
      V: [T,H/2,W/2]
    """
    w2, h2 = width // 2, height // 2
    y_n = width * height
    uv_n = w2 * h2

    Y = np.empty((num_frames, height, width), dtype=np.uint16)
    U = np.empty((num_frames, h2, w2), dtype=np.uint16)
    V = np.empty((num_frames, h2, w2), dtype=np.uint16)

    with open(path, "rb") as f:
        for t in range(num_frames):
            yb = f.read(y_n * 2)
            ub = f.read(uv_n * 2)
            vb = f.read(uv_n * 2)

            if len(yb) != y_n * 2 or len(ub) != uv_n * 2 or len(vb) != uv_n * 2:
                raise IOError(f"EOF while reading {path} at frame {t}")

            Y[t] = np.frombuffer(yb, dtype=np.dtype("<u2")).reshape(height, width)
            U[t] = np.frombuffer(ub, dtype=np.dtype("<u2")).reshape(h2, w2)
            V[t] = np.frombuffer(vb, dtype=np.dtype("<u2")).reshape(h2, w2)

    return Y, U, V


def write_yuv420p10le(
    path: Path,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    assert Y.dtype == np.uint16
    assert U.dtype == np.uint16
    assert V.dtype == np.uint16

    with open(path, "wb") as f:
        for t in range(Y.shape[0]):
            f.write(Y[t].astype("<u2", copy=False).tobytes(order="C"))
            f.write(U[t].astype("<u2", copy=False).tobytes(order="C"))
            f.write(V[t].astype("<u2", copy=False).tobytes(order="C"))


def to_float01_u10(x_u16: np.ndarray) -> np.ndarray:
    return x_u16.astype(np.float32) / 1023.0


def float01_to_u10(x01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x01, 0.0, 1.0) * 1023.0).astype(np.uint16)


# ============================================================
# Model loading
# ============================================================
def load_model(net_import: str, ckpt_path: str, device: torch.device, strict: bool = False):
    mod_name, cls_name = net_import.split(":")
    mod = __import__(mod_name, fromlist=[cls_name])
    NetCls = getattr(mod, cls_name)
    model = NetCls()

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=strict)
    model.to(device)
    model.eval()
    return model


def parse_ckpt_map(s: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    items = [x.strip() for x in s.split(",") if x.strip()]
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"Bad --ckpt_map item: {kv} (expected qp=/path)")
        k, v = kv.split("=", 1)
        out[int(k.strip())] = v.strip()
    if not out:
        raise ValueError("Empty --ckpt_map")
    return out


def pick_ckpt_for_qp(qp: int, ckpt_map: Dict[int, str], mode: str = "nearest") -> Tuple[int, str]:
    if mode == "exact":
        if qp not in ckpt_map:
            raise KeyError(f"no ckpt for qp={qp} (mode=exact)")
        return qp, ckpt_map[qp]
    keys = sorted(ckpt_map.keys())
    best = min(keys, key=lambda k: abs(k - qp))
    return best, ckpt_map[best]


# ============================================================
# Pad / crop helpers
# ============================================================
def pad_btchw_to_multiple(x: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    x: [B,T,C,H,W]
    """
    B, T, C, H, W = x.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    y = x.reshape(B * T, C, H, W)
    y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = y.shape[-2:]
    y = y.reshape(B, T, C, Hp, Wp)
    return y, (pad_h, pad_w)


def crop_btchw(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    return x[..., :x.shape[-2] - pad_h, :x.shape[-1] - pad_w]


# ============================================================
# 5-to-1 fullclip inference
# ============================================================
@torch.no_grad()
def run_net_5to1_fullclip(
    model,
    recY01: np.ndarray, recU01: np.ndarray, recV01: np.ndarray,  # (T,H,W)
    device: torch.device,
    amp: bool = False,
    batch: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    model expects:
      Yt: (B,5,1,H,W)
      Ut,Vt: (B,5,1,H/2,W/2)
    returns:
      eY: (T,H,W)
      eU,eV: (T,H/2,W/2)
    """
    T, H, W = recY01.shape
    H2, W2 = recU01.shape[1], recU01.shape[2]

    idx = np.arange(T)
    idx_pad = np.pad(idx, (2, 2), mode="edge")

    def make_stack(arr, t_center):
        ii = idx_pad[t_center: t_center + 5]
        return arr[ii]

    eY = np.empty((T, H, W), np.float32)
    eU = np.empty((T, H2, W2), np.float32)
    eV = np.empty((T, H2, W2), np.float32)

    t = 0
    while t < T:
        b = min(batch, T - t)

        Yb = np.stack([make_stack(recY01, tt) for tt in range(t, t + b)], axis=0)  # (B,5,H,W)
        Ub = np.stack([make_stack(recU01, tt) for tt in range(t, t + b)], axis=0)  # (B,5,H2,W2)
        Vb = np.stack([make_stack(recV01, tt) for tt in range(t, t + b)], axis=0)

        Yt = torch.from_numpy(Yb).to(device).unsqueeze(2)  # (B,5,1,H,W)
        Ut = torch.from_numpy(Ub).to(device).unsqueeze(2)
        Vt = torch.from_numpy(Vb).to(device).unsqueeze(2)

        Yt, (py, px) = pad_btchw_to_multiple(Yt, 16)
        Ut, (puy, pux) = pad_btchw_to_multiple(Ut, 16)
        Vt, (pvy, pvx) = pad_btchw_to_multiple(Vt, 16)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                oY, oU, oV = model(Yt, Ut, Vt)
        else:
            oY, oU, oV = model(Yt, Ut, Vt)

        # support output as (B,1,H,W) or (B,H,W)
        def _restore(o: torch.Tensor, pad_h: int, pad_w: int) -> np.ndarray:
            if o.dim() == 4:
                o = o[:, None, ...] if o.shape[1] != 1 else o
                o = o[:, None, 0] if o.dim() == 4 else o
            if o.dim() == 5:
                o = crop_btchw(o, pad_h, pad_w)
                o = o[:, 0, 0]  # (B,H,W)
            elif o.dim() == 4:
                # (B,1,H,W)
                o = o[:, :, :o.shape[-2] - pad_h if pad_h > 0 else o.shape[-2],
                         :o.shape[-1] - pad_w if pad_w > 0 else o.shape[-1]]
                o = o[:, 0]
            elif o.dim() == 3:
                pass
            else:
                raise RuntimeError(f"Unexpected output dims: {tuple(o.shape)}")
            return o.clamp(0, 1).float().cpu().numpy()

        oY = _restore(oY, py, px)
        oU = _restore(oU, puy, pux)
        oV = _restore(oV, pvy, pvx)

        eY[t:t + b] = oY
        eU[t:t + b] = oU
        eV[t:t + b] = oV
        t += b

    return eY, eU, eV


# ============================================================
# Path helpers
# ============================================================
def find_qp_dirs(codec_out: Path) -> List[Path]:
    return sorted([p for p in codec_out.iterdir() if p.is_dir() and p.name.startswith("qp")])


def parse_qp_from_dirname(name: str) -> int:
    if not name.startswith("qp"):
        raise ValueError(f"Bad qp dir name: {name}")
    return int(name[2:])


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
    ap.add_argument("--codec_out", type=str, required=True, help="codec output root containing qpXX/rec/...")

    ap.add_argument("--post_net", type=str, required=True, help="module.sub:ClassName")
    ap.add_argument("--post_ckpt_map", type=str, required=True, help="qp=/path,qp=/path,...")
    ap.add_argument("--ckpt_mode", type=str, default="nearest", choices=["nearest", "exact"])

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--batch", type=int, default=16)

    ap.add_argument("--only_seq", type=str, default="", help="comma-separated seq names")
    ap.add_argument("--frame_limit", type=int, default=0, help="0=use all frames from yaml, >0=use first N frames")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing rec_post files")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    codec_out = Path(args.codec_out)

    only_seq = {s.strip() for s in args.only_seq.split(",") if s.strip()}
    seq_items = collect_seq_items_from_yaml(
        Path(args.yaml),
        only_seq=only_seq if only_seq else None,
    )

    ckpt_map = parse_ckpt_map(args.post_ckpt_map)
    model_cache: Dict[str, torch.nn.Module] = {}

    qp_dirs = find_qp_dirs(codec_out)
    if not qp_dirs:
        raise RuntimeError(f"No qpXX dirs found under: {codec_out}")

    total = len(qp_dirs) * len(seq_items)
    done = 0

    for qp_dir in qp_dirs:
        qp = parse_qp_from_dirname(qp_dir.name)

        _, ckpt = pick_ckpt_for_qp(qp, ckpt_map, mode=args.ckpt_mode)
        if ckpt not in model_cache:
            model_cache[ckpt] = load_model(args.post_net, ckpt, device=device, strict=args.strict)
        model = model_cache[ckpt]

        for item in seq_items:
            seq_name = item["name"]
            seq_cls = item["seq_cls"]
            width = int(item["width"])
            height = int(item["height"])
            total_frames = int(item["frames"])

            used_frames = total_frames if args.frame_limit <= 0 else min(args.frame_limit, total_frames)

            rec_path = rec_path_for_seq(codec_out, qp, seq_cls, seq_name)
            out_path = rec_post_path_for_seq(codec_out, qp, seq_cls, seq_name)

            done += 1

            if not rec_path.is_file():
                print(f"[SKIP {done}/{total}] missing rec: {rec_path}")
                continue

            if out_path.is_file() and not args.overwrite:
                print(f"[SKIP {done}/{total}] exists: {out_path}")
                continue

            print(
                f"[{done}/{total}] postprocess "
                f"qp={qp} seq={seq_name} class={seq_cls} frames={used_frames}"
            )

            rY10, rU10, rV10 = read_yuv420p10le(
                rec_path,
                width=width,
                height=height,
                num_frames=used_frames,
            )

            rY = to_float01_u10(rY10)
            rU = to_float01_u10(rU10)
            rV = to_float01_u10(rV10)

            pY, pU, pV = run_net_5to1_fullclip(
                model,
                rY, rU, rV,
                device=device,
                amp=args.amp,
                batch=args.batch,
            )

            write_yuv420p10le(
                out_path,
                float01_to_u10(pY),
                float01_to_u10(pU),
                float01_to_u10(pV),
            )

    print("Done.")
    print(f"rec_post written under: {codec_out}")


if __name__ == "__main__":
    main()
