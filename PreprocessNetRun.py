#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import multiprocessing as mp

try:
    import yaml
except ImportError as e:
    raise ImportError("pyyaml is required. Install with: pip install pyyaml") from e


# ============================================================
# Fixed setup
# ============================================================
FRAMES = 33
_G = {}


# ============================================================
# YAML / cfg parsing
# ============================================================
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError("YAML root must be dict.")
    return d


def normalize_join(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def parse_seq_cfg(cfg_path: Path) -> Dict[str, str]:
    """
    First line ignored. From second line:
      Key : Value   # comment
    """
    txt = cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(txt) <= 1:
        return {}
    out: Dict[str, str] = {}
    for line in txt[1:]:
        line0 = line.split("#", 1)[0].strip()
        if not line0 or ":" not in line0:
            continue
        k, v = line0.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


# ============================================================
# YUV IO
# ============================================================
def _frame_layout(w: int, h: int):
    w2, h2 = w // 2, h // 2
    y_n = w * h
    uv_n = w2 * h2
    frame_n = y_n + 2 * uv_n
    return w2, h2, y_n, uv_n, frame_n


def open_yuv_memmap(path: Path, bitdepth: int, w: int, h: int, frames: int) -> np.memmap:
    """
    Return memmap 1D array of samples.
    - 8-bit  : uint8
    - 10-bit : little-endian uint16 container
    """
    w2, h2, y_n, uv_n, frame_n = _frame_layout(w, h)
    expected_samples = frame_n * frames

    if bitdepth <= 8:
        dtype = np.uint8
    else:
        dtype = np.dtype("<u2")

    bytes_per_sample = np.dtype(dtype).itemsize
    expected_bytes = expected_samples * bytes_per_sample
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"File too small: {path} expected>={expected_bytes} bytes for "
            f"{frames} frames, got {actual_bytes}"
        )

    return np.memmap(str(path), mode="r", dtype=dtype, shape=(expected_samples,))


def u8_to_u10_shift(x_u8: np.ndarray) -> np.ndarray:
    """
    8-bit -> 10-bit by left shift 2 bits.
    255 -> 1020
    """
    return x_u8.astype(np.uint16) << 2


def to_float01_from_u10(x_u10: np.ndarray) -> np.ndarray:
    return x_u10.astype(np.float32) / 1023.0


def float01_to_u10(x01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x01, 0.0, 1.0) * 1023.0).astype(np.uint16)


def extract_full_clip_u10(
    mm: np.memmap,
    bitdepth: int,
    w: int,
    h: int,
    frames: int = FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns uint16 10-bit arrays:
      Y10: (T,H,W), U10/V10: (T,H/2,W/2)
    If source is 8-bit, convert by <<2 so max is 1020.
    """
    w2, h2, y_n, uv_n, frame_n = _frame_layout(w, h)

    Y10 = np.empty((frames, h, w), dtype=np.uint16)
    U10 = np.empty((frames, h2, w2), dtype=np.uint16)
    V10 = np.empty((frames, h2, w2), dtype=np.uint16)

    for t in range(frames):
        base = t * frame_n
        y_base = base
        u_base = base + y_n
        v_base = base + y_n + uv_n

        for rr in range(h):
            row_start = y_base + rr * w
            row = mm[row_start: row_start + w]
            if bitdepth <= 8:
                Y10[t, rr] = u8_to_u10_shift(np.asarray(row, dtype=np.uint8))
            else:
                Y10[t, rr] = np.asarray(row, dtype=np.uint16)

        for rr in range(h2):
            u_row_start = u_base + rr * w2
            v_row_start = v_base + rr * w2
            u_row = mm[u_row_start: u_row_start + w2]
            v_row = mm[v_row_start: v_row_start + w2]
            if bitdepth <= 8:
                U10[t, rr] = u8_to_u10_shift(np.asarray(u_row, dtype=np.uint8))
                V10[t, rr] = u8_to_u10_shift(np.asarray(v_row, dtype=np.uint8))
            else:
                U10[t, rr] = np.asarray(u_row, dtype=np.uint16)
                V10[t, rr] = np.asarray(v_row, dtype=np.uint16)

    return Y10, U10, V10


def write_yuv420p10le(path: Path, Y10: np.ndarray, U10: np.ndarray, V10: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert Y10.dtype == np.uint16 and U10.dtype == np.uint16 and V10.dtype == np.uint16

    T, H, W = Y10.shape
    assert U10.shape == (T, H // 2, W // 2)
    assert V10.shape == (T, H // 2, W // 2)

    with open(path, "wb") as f:
        for t in range(T):
            f.write(Y10[t].astype("<u2", copy=False).tobytes(order="C"))
            f.write(U10[t].astype("<u2", copy=False).tobytes(order="C"))
            f.write(V10[t].astype("<u2", copy=False).tobytes(order="C"))


def write_meta_json(path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================
# Model / inference
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


@torch.no_grad()
def run_net_5to1_fullclip(
    model,
    recY01: np.ndarray, recU01: np.ndarray, recV01: np.ndarray,  # (T,H,W)
    device: torch.device,
    amp: bool = False,
    batch: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    model input:
      Yt: (B,5,1,H,W)
      Ut,Vt: (B,5,1,H/2,W/2)

    model output:
      oY,oU,oV each either (B,1,H,W) or (B,H,W)
    """
    T, H, W = recY01.shape
    H2, W2 = recU01.shape[1], recU01.shape[2]

    idx = np.arange(T)
    idx_pad = np.pad(idx, (2, 2), mode="edge")

    def make_stack(arr, t_center):
        ii = idx_pad[t_center: t_center + 5]
        return arr[ii]

    outY = np.empty((T, H, W), np.float32)
    outU = np.empty((T, H2, W2), np.float32)
    outV = np.empty((T, H2, W2), np.float32)

    t = 0
    while t < T:
        b = min(batch, T - t)

        Yb = np.stack([make_stack(recY01, tt) for tt in range(t, t + b)], axis=0)
        Ub = np.stack([make_stack(recU01, tt) for tt in range(t, t + b)], axis=0)
        Vb = np.stack([make_stack(recV01, tt) for tt in range(t, t + b)], axis=0)

        Yt = torch.from_numpy(Yb).to(device).unsqueeze(2)
        Ut = torch.from_numpy(Ub).to(device).unsqueeze(2)
        Vt = torch.from_numpy(Vb).to(device).unsqueeze(2)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                oY, oU, oV = model(Yt, Ut, Vt)
        else:
            oY, oU, oV = model(Yt, Ut, Vt)

        def _to_bhw(o: torch.Tensor) -> np.ndarray:
            if o.dim() == 4:
                o = o[:, 0]
            elif o.dim() != 3:
                raise RuntimeError(f"Unexpected output dims: {tuple(o.shape)}")
            return o.clamp(0, 1).float().cpu().numpy()

        outY[t:t + b] = _to_bhw(oY)
        outU[t:t + b] = _to_bhw(oU)
        outV[t:t + b] = _to_bhw(oV)
        t += b

    return outY, outU, outV


# ============================================================
# Worker
# ============================================================
def _worker_init(
    out_root_str: str,
    net_import: str,
    ckpt_path: str,
    device_str: str,
    strict: bool,
    amp: bool,
    batch: int,
):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out_root = Path(out_root_str)
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(net_import, ckpt_path, device=device, strict=strict)

    _G["out_root"] = out_root
    _G["device"] = device
    _G["model"] = model
    _G["amp"] = bool(amp)
    _G["batch"] = int(batch)


def process_one_sequence(job) -> Tuple[str, bool, str]:
    """
    job:
      (seq_name, seq_cls, yuv_path, w, h, bitdepth)
    """
    try:
        seq_name, seq_cls, yuv_path_str, w, h, bitdepth = job
        yuv_path = Path(yuv_path_str)

        out_root: Path = _G["out_root"]
        device: torch.device = _G["device"]
        model = _G["model"]
        amp: bool = _G["amp"]
        batch: int = _G["batch"]

        gt_dir = out_root / "gt_yuv"
        prep_dir = out_root / "preprocess_yuv"
        meta_root = out_root / "meta"

        clip_id = f"{seq_name}_t000"

        mm = open_yuv_memmap(yuv_path, bitdepth=bitdepth, w=w, h=h, frames=FRAMES)
        Y10, U10, V10 = extract_full_clip_u10(mm, bitdepth, w, h, frames=FRAMES)

        # save GT
        write_yuv420p10le(gt_dir / f"{clip_id}.yuv", Y10, U10, V10)

        meta_gt = {
            "clip_id": clip_id,
            "type": "gt",
            "seq_name": seq_name,
            "seq_cls": seq_cls,
            "src_yuv": str(yuv_path),
            "width": int(w),
            "height": int(h),
            "src_bitdepth": int(bitdepth),
            "frames": int(FRAMES),
            "format": "yuv420p10le",
        }
        write_meta_json(meta_root / "gt" / f"{clip_id}.json", meta_gt)

        # preprocess on GT
        Y01 = to_float01_from_u10(Y10)
        U01 = to_float01_from_u10(U10)
        V01 = to_float01_from_u10(V10)

        pY, pU, pV = run_net_5to1_fullclip(
            model,
            Y01, U01, V01,
            device=device,
            amp=amp,
            batch=batch,
        )

        pY10 = float01_to_u10(pY)
        pU10 = float01_to_u10(pU)
        pV10 = float01_to_u10(pV)

        write_yuv420p10le(prep_dir / f"{clip_id}.yuv", pY10, pU10, pV10)

        meta_prep = dict(meta_gt)
        meta_prep["type"] = "preprocess"
        meta_prep["preprocess_net"] = "applied"
        write_meta_json(meta_root / "preprocess" / f"{clip_id}.json", meta_prep)

        return (clip_id, True, "ok")

    except Exception as e:
        return ("?", False, f"err: {repr(e)}")


# ============================================================
# Main
# ============================================================
def build_preprocess_from_yaml(
    yaml_path: str,
    ra_cfg_path: str,
    out_root: str,
    net_import: str,
    ckpt_path: str,
    device: str = "cuda",
    strict: bool = False,
    amp: bool = False,
    batch: int = 8,
    num_workers: int = 4,
    chunksize: int = 1,
    max_seqs: int = 0,
):
    yml = load_yaml(Path(yaml_path))
    out_root = Path(out_root)

    if "path_seq" not in yml or "seq" not in yml:
        raise KeyError("YAML must contain keys: 'path_seq' and 'seq'")

    (out_root / "gt_yuv").mkdir(parents=True, exist_ok=True)
    (out_root / "preprocess_yuv").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "gt").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "preprocess").mkdir(parents=True, exist_ok=True)

    base_root = Path(yml["path_seq"])
    seq_dict: Dict = yml["seq"]
    seq_names = sorted(seq_dict.keys())
    if max_seqs and max_seqs > 0:
        seq_names = seq_names[:max_seqs]

    jobs: List[Tuple] = []

    for seq_name in seq_names:
        sd = seq_dict[seq_name]
        seq_cls = str(sd.get("seq_cls", "UNK"))
        seq_cfg_rel = sd.get("seq_cfg", None)
        if seq_cfg_rel is None:
            print(f"[SKIP] {seq_name}: missing seq_cfg")
            continue

        seq_cfg_path = normalize_join(base_root, str(seq_cfg_rel))
        if not seq_cfg_path.exists():
            print(f"[SKIP] {seq_name}: cfg not found: {seq_cfg_path}")
            continue

        cfg = parse_seq_cfg(seq_cfg_path)

        in_file = cfg.get("InputFile", None)
        if in_file is None:
            print(f"[SKIP] {seq_name}: cfg missing InputFile")
            continue

        yuv_path = normalize_join(seq_cfg_path.parent, in_file)
        if not yuv_path.exists():
            yuv_path2 = normalize_join(base_root, in_file)
            if yuv_path2.exists():
                yuv_path = yuv_path2
            else:
                print(f"[SKIP] {seq_name}: input yuv not found: {yuv_path}")
                continue

        bitdepth = int(cfg.get("InputBitDepth", "8"))
        w = int(cfg.get("SourceWidth", cfg.get("InputWidth", "0")))
        h = int(cfg.get("SourceHeight", cfg.get("InputHeight", "0")))
        if w <= 0 or h <= 0:
            print(f"[SKIP] {seq_name}: invalid resolution")
            continue

        jobs.append((seq_name, seq_cls, str(yuv_path), w, h, bitdepth))
        print(f"[SEQ] {seq_name} cls={seq_cls} {w}x{h} bd={bitdepth} input={yuv_path}")

    if not jobs:
        raise RuntimeError("No jobs to run. Check YAML/cfg paths and inputs.")

    run_info = {
        "yaml": str(yaml_path),
        "ra_cfg_path": str(ra_cfg_path),
        "out_root": str(out_root),
        "net_import": net_import,
        "ckpt_path": ckpt_path,
        "frames": FRAMES,
        "jobs": len(jobs),
    }
    write_meta_json(out_root / "run_info.json", run_info)

    print(f"[RUN] jobs={len(jobs)} workers={num_workers} chunksize={chunksize}")

    ctx = mp.get_context("spawn")
    ok = 0
    fail = 0

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(str(out_root), net_import, ckpt_path, device, strict, amp, batch),
    ) as pool:
        for i, (clip_id, is_ok, msg) in enumerate(pool.imap_unordered(process_one_sequence, jobs, chunksize=chunksize), 1):
            if is_ok:
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {clip_id}: {msg}")

            if i % 20 == 0 or i == len(jobs):
                print(f"progress {i}/{len(jobs)} (ok={ok}, fail={fail}) last={clip_id} {msg}")

    print("Done.")
    print(f"ok={ok}, fail={fail}")
    print("out_root:", out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, required=True, help="dataset yaml path")
    ap.add_argument("--ra_cfg", type=str, required=True, help="kept for pipeline consistency")
    ap.add_argument("--out_root", type=str, required=True, help="output root")
    ap.add_argument("--net_import", type=str, required=True, help="module.sub:ClassName")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    args = ap.parse_args()

    build_preprocess_from_yaml(
        yaml_path=args.yaml,
        ra_cfg_path=args.ra_cfg,
        out_root=args.out_root,
        net_import=args.net_import,
        ckpt_path=args.ckpt_path,
        device=args.device,
        strict=args.strict,
        amp=args.amp,
        batch=args.batch,
        num_workers=args.workers,
        chunksize=args.chunksize,
        max_seqs=args.max_seqs,
    )












#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shlex
import glob
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -------------------------
# bsub helper
# -------------------------
def submit_bsub_script(queue: str, job_name: str, script: str, dry_run: bool, ncore: int = 8, log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    bsub_args = [
        "bsub",
        "-q", queue,
        "-J", job_name,
        "-n", str(ncore),
        "-R", "span[hosts=1]",
        "-oo", f"{log_dir}/%J.out",
        "-eo", f"{log_dir}/%J.err",
        "/bin/bash",
    ]

    if dry_run:
        print("----- BSUB CMD -----")
        print(" ".join(shlex.quote(x) for x in bsub_args))
        print("----- SCRIPT  -----")
        print(script.rstrip())
        print("--------------------")
        return

    if not script.startswith("#!"):
        script = "#!/usr/bin/env bash\n" + script

    r = subprocess.run(
        bsub_args,
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    if r.returncode != 0:
        print("[BSUB FAILED]")
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        raise RuntimeError(f"bsub failed with returncode={r.returncode}")
    else:
        print(r.stdout.strip())


# -------------------------
# helpers
# -------------------------
def clip_id_from_yuv_path(p: str) -> str:
    return Path(p).stem


def find_meta_json(meta_root: Path, variant: str, clip_id: str) -> Optional[Path]:
    j = meta_root / variant / f"{clip_id}.json"
    return j if j.exists() else None


def load_meta(meta_path: Path) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def shlex_join(cmd_list: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd_list)


# -------------------------
# VTM command
# -------------------------
def build_encoder_cmd(
    encoder_app: str,
    cfg_path: str,
    in_yuv: str,
    out_bin: str,
    out_rec: str,
    wdt: int,
    hgt: int,
    bitdepth: int,
    internal_bd: int,
    framerate: int,
    frames: int,
    qp: int,
) -> str:
    cmd = [
        encoder_app,
        "-c", cfg_path,
        "-i", in_yuv,
        "-b", out_bin,
        "-o", out_rec,
        "-wdt", str(wdt),
        "-hgt", str(hgt),
        f"--InputBitDepth={bitdepth}",
        f"--OutputBitDepth={bitdepth}",
        f"--InternalBitDepth={internal_bd}",
        f"--FrameRate={framerate}",
        f"--FramesToBeEncoded={frames}",
        f"--QP={qp}",
    ]
    return shlex_join(cmd)


# -------------------------
# batching
# -------------------------
def chunk_list(xs: List[str], chunk: int) -> List[List[str]]:
    return [xs[i:i + chunk] for i in range(0, len(xs), chunk)]


def build_parallel_script(cmds: List[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -o pipefail",
        "",
        "pids=()",
        "rc=0",
        "",
    ]
    for i, c in enumerate(cmds, start=1):
        lines += [
            f'echo "[{i}/{len(cmds)}] {c}"',
            f"({c}) &",
            "pids+=( $! )",
            "",
        ]
    lines += [
        "for pid in \"${pids[@]}\"; do",
        "  if ! wait \"$pid\"; then",
        "    rc=1",
        "  fi",
        "done",
        "",
        "exit $rc",
    ]
    return "\n".join(lines)


# -------------------------
# main
# -------------------------
def submit_all_batched(
    data_root: str,
    encoder_app: str,
    cfg_path: str,
    queue: str = "normal",
    qps: Tuple[int, ...] = (22, 27, 32, 37),
    framerate: int = 30,
    frames: int = 33,
    bitdepth: int = 10,
    internal_bd: int = 10,
    dry_run: bool = True,
    only_variant: Optional[str] = None,   # gt / preprocess
    batch_size: int = 8,
    group_scope: str = "variant_qp",      # all / variant / variant_qp
    ncore: int = 8,
    log_dir: str = "logs",
):
    data_root = Path(data_root)
    meta_root = data_root / "meta"

    variants: List[Tuple[str, Path]] = [
        ("gt", data_root / "gt_yuv"),
        ("preprocess", data_root / "preprocess_yuv"),
    ]

    if only_variant is not None:
        variants = [(k, d) for (k, d) in variants if k == only_variant]
        if len(variants) == 0:
            raise ValueError(f"only_variant={only_variant} not found.")

    out_root = data_root / "vtm_ra"
    ensure_dir(out_root)

    skipped = 0
    all_cmds: List[Tuple[str, int, str]] = []

    for variant_key, in_dir in variants:
        yuv_files = sorted(glob.glob(str(in_dir / "*.yuv")))
        if len(yuv_files) == 0:
            print(f"[WARN] no yuv in {in_dir}")
            continue

        for in_yuv in yuv_files:
            clip_id = clip_id_from_yuv_path(in_yuv)

            meta_path = find_meta_json(meta_root, variant_key, clip_id)
            if meta_path is None:
                print(f"[SKIP] meta missing: variant={variant_key} clip={clip_id}")
                skipped += 1
                continue

            meta = load_meta(meta_path)
            wdt = int(meta["width"])
            hgt = int(meta["height"])

            for qp in qps:
                out_bin_dir = out_root / variant_key / f"qp{qp:02d}" / "bin"
                out_rec_dir = out_root / variant_key / f"qp{qp:02d}" / "rec"
                ensure_dir(out_bin_dir)
                ensure_dir(out_rec_dir)

                out_bin = str(out_bin_dir / f"{clip_id}.bin")
                out_rec = str(out_rec_dir / f"{clip_id}.yuv")

                cmd_str = build_encoder_cmd(
                    encoder_app=encoder_app,
                    cfg_path=cfg_path,
                    in_yuv=in_yuv,
                    out_bin=out_bin,
                    out_rec=out_rec,
                    wdt=wdt,
                    hgt=hgt,
                    bitdepth=bitdepth,
                    internal_bd=internal_bd,
                    framerate=framerate,
                    frames=frames,
                    qp=qp,
                )
                all_cmds.append((variant_key, qp, cmd_str))

    if group_scope == "all":
        groups = {"all": [c for _, _, c in all_cmds]}
    elif group_scope == "variant":
        groups: Dict[str, List[str]] = {}
        for vkey, _, c in all_cmds:
            groups.setdefault(vkey, []).append(c)
    elif group_scope == "variant_qp":
        groups: Dict[Tuple[str, int], List[str]] = {}
        for vkey, qp, c in all_cmds:
            groups.setdefault((vkey, qp), []).append(c)
    else:
        raise ValueError("group_scope must be 'variant_qp' or 'variant' or 'all'")

    total_jobs = 0
    total_cmds = 0

    for gkey, cmds in groups.items():
        if not cmds:
            continue
        chunks = chunk_list(cmds, batch_size)

        for j, chunk_cmds in enumerate(chunks, start=1):
            script = build_parallel_script(chunk_cmds)

            if isinstance(gkey, tuple):
                vkey, qp = gkey
                job_name = f"vtm_{vkey}_qp{qp:02d}_b{j:03d}"
            else:
                job_name = f"vtm_{gkey}_b{j:03d}"

            submit_bsub_script(
                queue=queue,
                job_name=job_name,
                script=script,
                dry_run=dry_run,
                ncore=ncore,
                log_dir=log_dir,
            )
            total_jobs += 1
            total_cmds += len(chunk_cmds)

    print(f"Done. jobs={total_jobs}, cmds={total_cmds}, skipped(meta-missing)={skipped}")
    print(f"Outputs under: {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--encoder_app", type=str, required=True)
    ap.add_argument("--cfg_path", type=str, required=True)
    ap.add_argument("--queue", type=str, default="normal")
    ap.add_argument("--qps", type=str, default="22,27,32,37")
    ap.add_argument("--framerate", type=int, default=30)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--bitdepth", type=int, default=10)
    ap.add_argument("--internal_bd", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only_variant", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_scope", type=str, default="variant_qp", choices=["all", "variant", "variant_qp"])
    ap.add_argument("--ncore", type=int, default=8)
    ap.add_argument("--log_dir", type=str, default="logs")
    args = ap.parse_args()

    qps = tuple(int(x.strip()) for x in args.qps.split(",") if x.strip())
    only_variant = args.only_variant.strip() or None

    submit_all_batched(
        data_root=args.data_root,
        encoder_app=args.encoder_app,
        cfg_path=args.cfg_path,
        queue=args.queue,
        qps=qps,
        framerate=args.framerate,
        frames=args.frames,
        bitdepth=args.bitdepth,
        internal_bd=args.internal_bd,
        dry_run=args.dry_run,
        only_variant=only_variant,
        batch_size=args.batch_size,
        group_scope=args.group_scope,
        ncore=args.ncore,
        log_dir=args.log_dir,
    )




