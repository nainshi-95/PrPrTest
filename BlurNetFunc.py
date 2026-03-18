import torch
import torch.nn.functional as F


def gaussian_kernel_bank_2d_fixed_k(
    sigma_min: float,
    sigma_max: float,
    sigma_num: int,
    k: int = 5,
    device="cpu",
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        sigma_values: [S]
        kernels:      [S, 1, k, k]
    """
    assert k % 2 == 1
    assert sigma_num >= 2

    sigma_values = torch.linspace(
        sigma_min, sigma_max, steps=sigma_num, device=device, dtype=dtype
    )  # [S]

    ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)  # [k]
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # [k, k]
    rr2 = xx * xx + yy * yy  # [k, k]

    # sigma_values: [S] -> [S,1,1]
    sigma = sigma_values.view(-1, 1, 1)  # [S,1,1]

    # Handle sigma <= 0 as identity kernel
    # Usually sigma_min > 0, but keep safe.
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    safe_sigma = torch.clamp(sigma, min=eps)

    kernels = torch.exp(-rr2.unsqueeze(0) / (2.0 * safe_sigma * safe_sigma))  # [S,k,k]
    kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True)  # [S,k,k]

    if (sigma_values <= 0).any():
        zero_mask = (sigma_values <= 0)
        kernels[zero_mask] = 0
        kernels[zero_mask, k // 2, k // 2] = 1.0

    kernels = kernels.unsqueeze(1)  # [S,1,k,k]
    return sigma_values, kernels


def blur_bt1hw_sigma_palette(
    x_bt1hw: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    sigma_num: int,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blur input with all sigma values in parallel.

    Args:
        x_bt1hw: [B, T, 1, H, W]

    Returns:
        sigma_values: [S]
        palette_btshw: [B, T, S, H, W]
    """
    if x_bt1hw.ndim != 5:
        raise ValueError(f"x_bt1hw must be [B,T,1,H,W], got {tuple(x_bt1hw.shape)}")
    B, T, C, H, W = x_bt1hw.shape
    if C != 1:
        raise ValueError(f"Expected C=1, got C={C}")

    device = x_bt1hw.device
    dtype = x_bt1hw.dtype

    sigma_values, kernels = gaussian_kernel_bank_2d_fixed_k(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_num=sigma_num,
        k=k,
        device=device,
        dtype=dtype,
    )  # sigma_values:[S], kernels:[S,1,k,k]

    x_2d = x_bt1hw.reshape(B * T, 1, H, W)  # [BT,1,H,W]
    pad = k // 2
    x_pad = F.pad(x_2d, (pad, pad, pad, pad), mode="reflect")

    # One conv -> all sigma outputs in parallel
    # weight [S,1,k,k], input [BT,1,H,W] -> output [BT,S,H,W]
    y_2d = F.conv2d(x_pad, kernels, bias=None, stride=1, padding=0)

    palette_btshw = y_2d.view(B, T, sigma_num, H, W)  # [B,T,S,H,W]
    return sigma_values, palette_btshw


def make_triangular_interp_weights(
    strength_bt1hw: torch.Tensor,
    sigma_num: int,
) -> torch.Tensor:
    """
    Convert strength map in [0,1] to linear-interpolation weights over sigma palette.

    This is equivalent to interpolating between the two nearest sigma bins, but done
    in a differentiable way without integer indexing.

    Args:
        strength_bt1hw: [B, T, 1, H, W] in [0,1]
        sigma_num: number of sigma bins

    Returns:
        weights_btshw: [B, T, S, H, W]
    """
    if strength_bt1hw.ndim != 5:
        raise ValueError(
            f"strength_bt1hw must be [B,T,1,H,W], got {tuple(strength_bt1hw.shape)}"
        )
    if sigma_num < 2:
        raise ValueError("sigma_num must be >= 2")

    B, T, C, H, W = strength_bt1hw.shape
    if C != 1:
        raise ValueError(f"Expected C=1 for strength map, got C={C}")

    strength = strength_bt1hw.clamp(0.0, 1.0)  # [B,T,1,H,W]
    pos = strength * float(sigma_num - 1)      # continuous palette coordinate

    # Palette indices 0..S-1
    idx = torch.arange(
        sigma_num,
        device=strength.device,
        dtype=strength.dtype,
    ).view(1, 1, sigma_num, 1, 1)  # [1,1,S,1,1]

    # Triangular basis:
    # weight_j = max(1 - |pos - j|, 0)
    # This gives exactly two non-zero neighbors for interior points.
    weights = torch.relu(1.0 - torch.abs(pos - idx))  # [B,T,S,H,W]

    # Normalize just in case of tiny numerical issues
    weights = weights / torch.clamp(weights.sum(dim=2, keepdim=True), min=1e-12)
    return weights


def blur_bt1hw_from_strength_map(
    x_bt1hw: torch.Tensor,
    strength_bt1hw: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    sigma_num: int,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable w.r.t. BOTH:
      - x_bt1hw
      - strength_bt1hw

    Args:
        x_bt1hw:        [B, T, 1, H, W]
        strength_bt1hw: [B, T, 1, H, W], in [0,1]
        sigma_min:
        sigma_max:
        sigma_num:
        k:

    Returns:
        sigma_values:    [S]
        palette_btshw:   [B, T, S, H, W]
        weights_btshw:   [B, T, S, H, W]
        y_bt1hw:         [B, T, 1, H, W]
    """
    if x_bt1hw.shape != strength_bt1hw.shape:
        raise ValueError(
            f"x and strength must have same shape, got "
            f"{tuple(x_bt1hw.shape)} vs {tuple(strength_bt1hw.shape)}"
        )

    sigma_values, palette_btshw = blur_bt1hw_sigma_palette(
        x_bt1hw=x_bt1hw,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_num=sigma_num,
        k=k,
    )  # [B,T,S,H,W]

    weights_btshw = make_triangular_interp_weights(
        strength_bt1hw=strength_bt1hw,
        sigma_num=sigma_num,
    )  # [B,T,S,H,W]

    # Weighted sum over sigma dimension
    y_bt1hw = (palette_btshw * weights_btshw).sum(dim=2, keepdim=True)  # [B,T,1,H,W]

    return sigma_values, palette_btshw, weights_btshw, y_bt1hw











import torch
import torch.nn.functional as F


def hadamard_matrix(n: int, device, dtype):
    """
    Sylvester construction.
    n must be power of 2.
    Returns [n, n].
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be power of 2, got {n}")

    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [
                torch.cat([H,  H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    return H


def satd_16x16_btchw(res_btchw: torch.Tensor) -> torch.Tensor:
    """
    Compute mean 16x16 block SATD over residual tensor.

    Args:
        res_btchw: [B, T, C, H, W], usually C=1

    Returns:
        scalar tensor
    """
    if res_btchw.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(res_btchw.shape)}")

    B, T, C, H, W = res_btchw.shape
    if C != 1:
        raise ValueError(f"Expected C=1, got C={C}")

    blk = 16
    if H < blk or W < blk:
        raise ValueError(f"H and W must be >= 16, got H={H}, W={W}")

    # crop to multiple of 16
    Hc = (H // blk) * blk
    Wc = (W // blk) * blk
    x = res_btchw[:, :, :, :Hc, :Wc]  # [B,T,1,Hc,Wc]

    # [B*T, 1, Hc, Wc]
    x = x.reshape(B * T, 1, Hc, Wc)

    # unfold into 16x16 non-overlapping blocks
    # -> [BT, 16*16, Nblk]
    patches = F.unfold(x, kernel_size=blk, stride=blk)

    # -> [BT, Nblk, 16, 16]
    patches = patches.transpose(1, 2).reshape(-1, blk, blk)

    H16 = hadamard_matrix(blk, device=patches.device, dtype=patches.dtype)  # [16,16]

    # 2D Hadamard transform: H * X * H^T
    # Since H is symmetric, H^T = H
    coeff = H16 @ patches @ H16

    # SATD = sum(abs(coeff)) / scale
    # scale factor is convention-dependent; constant factor does not matter for loss.
    satd = coeff.abs().sum(dim=(1, 2)) / blk  # [BT * Nblk]

    return satd.mean()


def temporal_consistency_satd_loss(
    blur_bt1hw: torch.Tensor,
    spynet,
    warp_fn,
    detach_flow: bool = False,
    return_details: bool = False,
):
    """
    Temporal consistency loss using forward/backward motion compensation + 16x16 SATD.

    Args:
        blur_bt1hw: [B, T, 1, H, W]
        spynet:
            A flow network callable like:
                flow = spynet(img1, img2)
            where img1,img2 are [B,1,H,W] or [B,C,H,W]
            and flow is [B,2,H,W].

            Assumption:
              - spynet(a, b) returns flow that warps 'a' toward 'b'
                OR at least is consistent with warp_fn below.
        warp_fn:
            A warping function callable like:
                warped = warp_fn(src, flow)
            where src is [B,1,H,W], flow is [B,2,H,W],
            and output is [B,1,H,W].

            IMPORTANT:
              This code assumes:
                warp_fn(frame_{t+1}, flow_{t+1->t}) predicts frame_t
                warp_fn(frame_{t},   flow_{t->t+1}) predicts frame_{t+1}

        detach_flow:
            If True, flow is detached before warping.
            Useful when you do not want gradients into SPyNet.
        return_details:
            If True, returns a dict with intermediate losses.

    Returns:
        loss (scalar tensor)
        or
        (loss, details)
    """
    if blur_bt1hw.ndim != 5:
        raise ValueError(f"Expected blur_bt1hw [B,T,1,H,W], got {tuple(blur_bt1hw.shape)}")

    B, T, C, H, W = blur_bt1hw.shape
    if C != 1:
        raise ValueError(f"Expected C=1, got C={C}")
    if T < 2:
        raise ValueError(f"T must be >= 2, got T={T}")

    # Collect residual tensors from:
    # 1) next -> current  (backward flow)
    # 2) prev -> current  (forward flow)
    residuals = []

    loss_next_to_curr = []
    loss_prev_to_curr = []

    for t in range(T - 1):
        cur = blur_bt1hw[:, t]       # [B,1,H,W]
        nxt = blur_bt1hw[:, t + 1]   # [B,1,H,W]

        # flow from current to next
        flow_t_to_t1 = spynet(cur, nxt)      # [B,2,H,W]
        # flow from next to current
        flow_t1_to_t = spynet(nxt, cur)      # [B,2,H,W]

        if detach_flow:
            flow_t_to_t1 = flow_t_to_t1.detach()
            flow_t1_to_t = flow_t1_to_t.detach()

        # Predict current frame from next frame
        pred_cur_from_next = warp_fn(nxt, flow_t1_to_t)   # should align nxt -> cur
        res_cur_from_next = cur - pred_cur_from_next      # [B,1,H,W]
        residuals.append(res_cur_from_next.unsqueeze(1))  # [B,1,1,H,W]

        # Predict next frame from current frame
        pred_next_from_cur = warp_fn(cur, flow_t_to_t1)   # should align cur -> nxt
        res_next_from_cur = nxt - pred_next_from_cur      # [B,1,H,W]
        residuals.append(res_next_from_cur.unsqueeze(1))  # [B,1,1,H,W]

        # optional per-direction bookkeeping
        loss_next_to_curr.append(res_cur_from_next.unsqueeze(1))
        loss_prev_to_curr.append(res_next_from_cur.unsqueeze(1))

    # residual list elements are [B,1,1,H,W]
    # concatenate along T-like dimension
    residuals_bt1hw = torch.cat(residuals, dim=1)  # [B, 2*(T-1), 1, H, W]

    total_loss = satd_16x16_btchw(residuals_bt1hw)

    if not return_details:
        return total_loss

    next_to_curr_bt1hw = torch.cat(loss_next_to_curr, dim=1)  # [B,T-1,1,H,W]
    prev_to_curr_bt1hw = torch.cat(loss_prev_to_curr, dim=1)  # [B,T-1,1,H,W]

    details = {
        "loss_total": total_loss,
        "loss_next_to_curr": satd_16x16_btchw(next_to_curr_bt1hw),
        "loss_prev_to_curr": satd_16x16_btchw(prev_to_curr_bt1hw),
        "num_pairs": T - 1,
        "num_residual_frames": 2 * (T - 1),
    }
    return total_loss, details


