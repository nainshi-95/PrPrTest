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
