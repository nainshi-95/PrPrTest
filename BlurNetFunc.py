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

    Hc = (H // blk) * blk
    Wc = (W // blk) * blk
    x = res_btchw[:, :, :, :Hc, :Wc]  # [B,T,1,Hc,Wc]

    x = x.reshape(B * T, 1, Hc, Wc)  # [BT,1,Hc,Wc]

    # [BT, 16*16, Nblk]
    patches = F.unfold(x, kernel_size=blk, stride=blk)

    # [BT, Nblk, 16, 16]
    patches = patches.transpose(1, 2).reshape(-1, blk, blk)

    H16 = hadamard_matrix(blk, device=patches.device, dtype=patches.dtype)  # [16,16]

    coeff = H16 @ patches @ H16
    satd = coeff.abs().sum(dim=(1, 2)) / blk

    return satd.mean()


def resize_flow(flow_b2hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    Resize flow to new spatial size and scale displacement accordingly.

    Args:
        flow_b2hw: [B,2,H,W]
    Returns:
        resized flow: [B,2,out_h,out_w]
    """
    if flow_b2hw.ndim != 4 or flow_b2hw.shape[1] != 2:
        raise ValueError(f"Expected flow [B,2,H,W], got {tuple(flow_b2hw.shape)}")

    B, _, H, W = flow_b2hw.shape
    if H == out_h and W == out_w:
        return flow_b2hw

    flow_resized = F.interpolate(
        flow_b2hw,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )

    scale_x = float(out_w) / float(W)
    scale_y = float(out_h) / float(H)

    flow_resized[:, 0:1] *= scale_x
    flow_resized[:, 1:2] *= scale_y
    return flow_resized


def temporal_consistency_satd_loss_yuv420(
    y_bt1hw: torch.Tensor,
    u_bt1hw: torch.Tensor,
    v_bt1hw: torch.Tensor,
    spynet,
    warp_fn,
    detach_flow: bool = False,
    y_weight: float = 1.0,
    u_weight: float = 1.0,
    v_weight: float = 1.0,
    return_details: bool = False,
):
    """
    Temporal consistency loss for YUV420 inputs.

    Flow is computed ONCE using YUV444 made by bilinear upsampling U/V.

    Inputs:
        y_bt1hw: [B,T,1,H,W]
        u_bt1hw: [B,T,1,H/2,W/2]
        v_bt1hw: [B,T,1,H/2,W/2]

    Flow strategy:
        1) Upsample U,V to Y resolution
        2) Concatenate Y,U_up,V_up -> YUV444 [B,T,3,H,W]
        3) Compute forward/backward flow once on YUV444
        4) Use original flow for Y
        5) Resize flow to UV resolution for U/V and scale displacement accordingly

    Warp convention assumption:
        - spynet(a, b) returns flow mapping a -> b
        - warp_fn(src, flow_ab) warps src into b-domain

    Returns:
        scalar loss
        or (loss, details)
    """
    # -----------------------------
    # shape checks
    # -----------------------------
    if y_bt1hw.ndim != 5 or u_bt1hw.ndim != 5 or v_bt1hw.ndim != 5:
        raise ValueError("All inputs must be 5D tensors")

    By, Ty, Cy, Hy, Wy = y_bt1hw.shape
    Bu, Tu, Cu, Hu, Wu = u_bt1hw.shape
    Bv, Tv, Cv, Hv, Wv = v_bt1hw.shape

    if Cy != 1 or Cu != 1 or Cv != 1:
        raise ValueError("Expected single-channel Y/U/V tensors")

    if not (By == Bu == Bv and Ty == Tu == Tv):
        raise ValueError("Y/U/V must have same batch/time dimensions")

    if Hu * 2 != Hy or Wu * 2 != Wy or Hv * 2 != Hy or Wv * 2 != Wy:
        raise ValueError(
            f"Expected YUV420 shapes, got Y={tuple(y_bt1hw.shape)}, "
            f"U={tuple(u_bt1hw.shape)}, V={tuple(v_bt1hw.shape)}"
        )

    B, T = By, Ty
    if T < 2:
        raise ValueError(f"T must be >= 2, got T={T}")

    # -----------------------------
    # build YUV444 for flow
    # -----------------------------
    u_up = F.interpolate(
        u_bt1hw.reshape(B * T, 1, Hu, Wu),
        size=(Hy, Wy),
        mode="bilinear",
        align_corners=False,
    ).view(B, T, 1, Hy, Wy)

    v_up = F.interpolate(
        v_bt1hw.reshape(B * T, 1, Hv, Wv),
        size=(Hy, Wy),
        mode="bilinear",
        align_corners=False,
    ).view(B, T, 1, Hy, Wy)

    yuv444_bt3hw = torch.cat([y_bt1hw, u_up, v_up], dim=2)  # [B,T,3,H,W]

    # -----------------------------
    # collect residuals
    # -----------------------------
    y_residuals = []
    u_residuals = []
    v_residuals = []

    y_next_to_curr = []
    y_prev_to_next = []
    u_next_to_curr = []
    u_prev_to_next = []
    v_next_to_curr = []
    v_prev_to_next = []

    for t in range(T - 1):
        cur_444 = yuv444_bt3hw[:, t]      # [B,3,Hy,Wy]
        nxt_444 = yuv444_bt3hw[:, t + 1]  # [B,3,Hy,Wy]

        # flows on YUV444
        flow_t_to_t1 = spynet(cur_444, nxt_444)     # [B,2,Hy,Wy]
        flow_t1_to_t = spynet(nxt_444, cur_444)     # [B,2,Hy,Wy]

        if detach_flow:
            flow_t_to_t1 = flow_t_to_t1.detach()
            flow_t1_to_t = flow_t1_to_t.detach()

        # -------------------------
        # Y residuals (full-res flow)
        # -------------------------
        y_cur = y_bt1hw[:, t]       # [B,1,Hy,Wy]
        y_nxt = y_bt1hw[:, t + 1]   # [B,1,Hy,Wy]

        y_pred_cur_from_next = warp_fn(y_nxt, flow_t1_to_t)
        y_res_cur_from_next = y_cur - y_pred_cur_from_next

        y_pred_next_from_cur = warp_fn(y_cur, flow_t_to_t1)
        y_res_next_from_cur = y_nxt - y_pred_next_from_cur

        y_residuals.append(y_res_cur_from_next.unsqueeze(1))
        y_residuals.append(y_res_next_from_cur.unsqueeze(1))

        y_next_to_curr.append(y_res_cur_from_next.unsqueeze(1))
        y_prev_to_next.append(y_res_next_from_cur.unsqueeze(1))

        # -------------------------
        # UV residuals (half-res flow)
        # -------------------------
        flow_t_to_t1_uv = resize_flow(flow_t_to_t1, Hu, Wu)
        flow_t1_to_t_uv = resize_flow(flow_t1_to_t, Hu, Wu)

        u_cur = u_bt1hw[:, t]       # [B,1,Hu,Wu]
        u_nxt = u_bt1hw[:, t + 1]
        v_cur = v_bt1hw[:, t]
        v_nxt = v_bt1hw[:, t + 1]

        u_pred_cur_from_next = warp_fn(u_nxt, flow_t1_to_t_uv)
        u_res_cur_from_next = u_cur - u_pred_cur_from_next

        u_pred_next_from_cur = warp_fn(u_cur, flow_t_to_t1_uv)
        u_res_next_from_cur = u_nxt - u_pred_next_from_cur

        v_pred_cur_from_next = warp_fn(v_nxt, flow_t1_to_t_uv)
        v_res_cur_from_next = v_cur - v_pred_cur_from_next

        v_pred_next_from_cur = warp_fn(v_cur, flow_t_to_t1_uv)
        v_res_next_from_cur = v_nxt - v_pred_next_from_cur

        u_residuals.append(u_res_cur_from_next.unsqueeze(1))
        u_residuals.append(u_res_next_from_cur.unsqueeze(1))
        v_residuals.append(v_res_cur_from_next.unsqueeze(1))
        v_residuals.append(v_res_next_from_cur.unsqueeze(1))

        u_next_to_curr.append(u_res_cur_from_next.unsqueeze(1))
        u_prev_to_next.append(u_res_next_from_cur.unsqueeze(1))
        v_next_to_curr.append(v_res_cur_from_next.unsqueeze(1))
        v_prev_to_next.append(v_res_next_from_cur.unsqueeze(1))

    # [B,2*(T-1),1,H,W] for Y
    y_residuals_bt1hw = torch.cat(y_residuals, dim=1)
    u_residuals_bt1hw = torch.cat(u_residuals, dim=1)
    v_residuals_bt1hw = torch.cat(v_residuals, dim=1)

    y_loss = satd_16x16_btchw(y_residuals_bt1hw)
    u_loss = satd_16x16_btchw(u_residuals_bt1hw)
    v_loss = satd_16x16_btchw(v_residuals_bt1hw)

    total_weight = y_weight + u_weight + v_weight
    total_loss = (y_weight * y_loss + u_weight * u_loss + v_weight * v_loss) / total_weight

    if not return_details:
        return total_loss

    details = {
        "loss_total": total_loss,
        "loss_y": y_loss,
        "loss_u": u_loss,
        "loss_v": v_loss,
        "num_pairs": T - 1,
        "num_residual_frames_per_channel": 2 * (T - 1),

        "loss_y_next_to_curr": satd_16x16_btchw(torch.cat(y_next_to_curr, dim=1)),
        "loss_y_prev_to_next": satd_16x16_btchw(torch.cat(y_prev_to_next, dim=1)),
        "loss_u_next_to_curr": satd_16x16_btchw(torch.cat(u_next_to_curr, dim=1)),
        "loss_u_prev_to_next": satd_16x16_btchw(torch.cat(u_prev_to_next, dim=1)),
        "loss_v_next_to_curr": satd_16x16_btchw(torch.cat(v_next_to_curr, dim=1)),
        "loss_v_prev_to_next": satd_16x16_btchw(torch.cat(v_prev_to_next, dim=1)),
    }
    return total_loss, details
