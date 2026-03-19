import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# ============================================================
# File utils
# ============================================================
def find_npz_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))


def load_npz_yuv(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    if "Y" not in data or "U" not in data or "V" not in data:
        raise KeyError(f"{npz_path} must contain keys: Y, U, V")

    Y = data["Y"]
    U = data["U"]
    V = data["V"]

    if Y.ndim != 3 or U.ndim != 3 or V.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays [T,H,W], got Y={Y.shape}, U={U.shape}, V={V.shape}"
        )

    return Y, U, V


def to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 1023.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def float01_to_uint10(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.round(arr * 1023.0).astype(np.uint16)
    return arr


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

    T = Y.shape[0]
    with open(out_path, "wb") as f:
        for t in range(T):
            Y[t].astype("<u2").tofile(f)
            U[t].astype("<u2").tofile(f)
            V[t].astype("<u2").tofile(f)


# ============================================================
# Model
# ============================================================
def build_model():
    """
    Replace this with your actual blur network creation/loading.
    Expected:
        y_out, u_out, v_out = model(y, u, v)

    Input:
        y: [B,T,1,H,W]
        u: [B,T,1,H/2,W/2]
        v: [B,T,1,H/2,W/2]

    Output:
        same shapes
    """
    raise NotImplementedError("Replace build_model() with your actual model loader.")


# ============================================================
# Inference helpers
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
        Y3: [3,H,W] float in [0,1]
        U3: [3,H/2,W/2] float in [0,1]
        V3: [3,H/2,W/2] float in [0,1]

    Returns:
        Y_out: [3,H,W]
        U_out: [3,H/2,W/2]
        V_out: [3,H/2,W/2]
    """
    y = torch.from_numpy(Y3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
    u = torch.from_numpy(U3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H/2,W/2]
    v = torch.from_numpy(V3).to(device).unsqueeze(0).unsqueeze(2)  # [1,3,1,H/2,W/2]

    y_out, u_out, v_out = model(y, u, v)

    # expected [1,3,1,H,W] etc
    y_out = y_out.squeeze(0).squeeze(1).detach().cpu().numpy()  # [3,H,W]
    u_out = u_out.squeeze(0).squeeze(1).detach().cpu().numpy()  # [3,H/2,W/2]
    v_out = v_out.squeeze(0).squeeze(1).detach().cpu().numpy()  # [3,H/2,W/2]

    return y_out, u_out, v_out


@torch.no_grad()
def process_one_npz(
    model: torch.nn.Module,
    npz_path: Path,
    out_yuv_path: Path,
    device: torch.device,
):
    """
    Requirement:
      original Vimeo sequence has 7 frames
      use only first two triplets:
        [0,1,2], [3,4,5]
      output total 6 frames
    """
    Y, U, V = load_npz_yuv(npz_path)

    Y = to_float01(Y)
    U = to_float01(U)
    V = to_float01(V)

    T = Y.shape[0]
    if T < 6:
        raise ValueError(f"{npz_path} has only {T} frames, but need at least 6")

    Y_012, U_012, V_012 = Y[0:3], U[0:3], V[0:3]
    Y_345, U_345, V_345 = Y[3:6], U[3:6], V[3:6]

    Y_out_012, U_out_012, V_out_012 = run_model_on_triplet(
        model, Y_012, U_012, V_012, device
    )
    Y_out_345, U_out_345, V_out_345 = run_model_on_triplet(
        model, Y_345, U_345, V_345, device
    )

    Y_out = np.concatenate([Y_out_012, Y_out_345], axis=0)  # [6,H,W]
    U_out = np.concatenate([U_out_012, U_out_345], axis=0)  # [6,H/2,W/2]
    V_out = np.concatenate([V_out_012, V_out_345], axis=0)  # [6,H/2,W/2]

    Y_u10 = float01_to_uint10(Y_out)
    U_u10 = float01_to_uint10(U_out)
    V_u10 = float01_to_uint10(V_out)

    save_yuv420p10le(out_yuv_path, Y_u10, U_u10, V_u10)


# ============================================================
# Encoder command helpers
# ============================================================
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
) -> str:
    """
    Adjust this function if your EncoderApp CLI differs.

    Current example is VTM/EncoderApp-like.
    stdout/stderr are redirected to log file.
    """
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
    """
    Submit:
      bsub -J JOB "(cmd1)& (cmd2)& ... & wait"
    """
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


def chunk_list(xs: List, chunk_size: int) -> List[List]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # blur inference
    parser.add_argument("--npz_root", type=str, required=True, help="Root folder containing .npz files")
    parser.add_argument("--blur_out_root", type=str, required=True, help="Where blurred .yuv files are saved")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # encoder submit
    parser.add_argument("--submit_bsub", action="store_true", help="Submit EncoderApp jobs with bsub")
    parser.add_argument("--cfg_path", type=str, default="", help="Encoder config file path")
    parser.add_argument("--bin_dir", type=str, default="", help="Folder containing EncoderApp")
    parser.add_argument("--encoder_name", type=str, default="EncoderApp", help="Encoder executable name inside bin_dir")
    parser.add_argument("--bit_out_root", type=str, default="", help="Bitstream output root")
    parser.add_argument("--recon_out_root", type=str, default="", help="Recon output root")
    parser.add_argument("--log_out_root", type=str, default="", help="Encoder log output root")

    parser.add_argument("--batch_size", type=int, default=8, help="How many sequences per one bsub submission")
    parser.add_argument("--job_prefix", type=str, default="HEVC_8CPU", help="bsub job name prefix")
    parser.add_argument("--queue", type=str, default="", help="Optional bsub queue")
    parser.add_argument("--extra_bsub_args", type=str, default="", help='Extra args for bsub, e.g. \'-n 8 -R "span[hosts=1]"\'')
    parser.add_argument("--dry_run_bsub", action="store_true", help="Print bsub command only")

    # encoder params
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--qp", type=int, default=32)
    parser.add_argument("--intra_period", type=int, default=32)
    parser.add_argument("--frame_rate", type=int, default=30)

    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    blur_out_root = Path(args.blur_out_root)
    device = torch.device(args.device)

    npz_files = find_npz_files(npz_root)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {npz_root}")

    model = build_model()
    model = model.to(device)
    model.eval()

    generated_yuvs: List[Path] = []

    # ------------------------------------------------------------
    # Step 1: npz -> blurred yuv
    # ------------------------------------------------------------
    for i, npz_path in enumerate(npz_files, 1):
        rel = npz_path.relative_to(npz_root)
        out_yuv_path = (blur_out_root / rel).with_suffix(".yuv")

        print(f"[{i}/{len(npz_files)}] Blur infer: {npz_path}")
        process_one_npz(
            model=model,
            npz_path=npz_path,
            out_yuv_path=out_yuv_path,
            device=device,
        )
        generated_yuvs.append(out_yuv_path)

    print(f"[INFO] Blurred YUVs saved under: {blur_out_root}")

    # ------------------------------------------------------------
    # Step 2: bsub submit EncoderApp
    # ------------------------------------------------------------
    if not args.submit_bsub:
        return

    if not args.cfg_path or not args.bin_dir or not args.bit_out_root or not args.recon_out_root or not args.log_out_root:
        raise ValueError(
            "When --submit_bsub is used, you must provide "
            "--cfg_path, --bin_dir, --bit_out_root, --recon_out_root, --log_out_root"
        )

    cfg_path = Path(args.cfg_path)
    encoder_app = Path(args.bin_dir) / args.encoder_name
    bit_out_root = Path(args.bit_out_root)
    recon_out_root = Path(args.recon_out_root)
    log_out_root = Path(args.log_out_root)

    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg_path not found: {cfg_path}")
    if not encoder_app.exists():
        raise FileNotFoundError(f"EncoderApp not found: {encoder_app}")

    cmd_list = []
    for yuv_path in generated_yuvs:
        rel = yuv_path.relative_to(blur_out_root).with_suffix("")

        bit_path = (bit_out_root / rel).with_suffix(".bin")
        recon_path = (recon_out_root / rel).with_suffix(".yuv")
        log_path = (log_out_root / rel).with_suffix(".log")

        bit_path.parent.mkdir(parents=True, exist_ok=True)
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_encoder_cmd(
            encoder_app=encoder_app,
            cfg_path=cfg_path,
            input_yuv=yuv_path,
            bitstream_path=bit_path,
            recon_path=recon_path,
            log_path=log_path,
            width=args.width,
            height=args.height,
            frames=args.frames,
            qp=args.qp,
            intra_period=args.intra_period,
            frame_rate=args.frame_rate,
        )
        cmd_list.append(cmd)

    batches = chunk_list(cmd_list, args.batch_size)

    for bi, batch_cmds in enumerate(batches):
        job_name = f"{args.job_prefix}_{bi:04d}"
        submit_bsub_batch(
            cmds=batch_cmds,
            job_name=job_name,
            queue=args.queue,
            extra_bsub_args=args.extra_bsub_args,
            dry_run=args.dry_run_bsub,
        )

    print(f"[INFO] Submitted {len(batches)} bsub jobs")
    print(f"[INFO] Bitstreams under: {bit_out_root}")
    print(f"[INFO] Recons under: {recon_out_root}")
    print(f"[INFO] Logs under: {log_out_root}")


if __name__ == "__main__":
    main()
