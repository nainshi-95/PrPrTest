import torch
import torch.nn.functional as F


def gaussian_kernel_2d_fixed_k(
    sigma: float,
    k: int = 5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns:
        ker: [1, 1, k, k]
    """
    assert k % 2 == 1
    if sigma <= 0:
        ker = torch.zeros((1, 1, k, k), device=device, dtype=dtype)
        ker[..., k // 2, k // 2] = 1.0
        return ker

    ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker = ker / ker.sum()
    return ker.view(1, 1, k, k)


def build_sigma_palette(
    sigma_min: float,
    sigma_max: float,
    sigma_num: int,
    k: int,
    device,
    dtype,
):
    """
    Returns:
        sigma_values: [S]
        kernels:      [S, 1, k, k]
    """
    if sigma_num < 2:
        raise ValueError("sigma_num must be >= 2")

    sigma_values = torch.linspace(
        sigma_min,
        sigma_max,
        steps=sigma_num,
        device=device,
        dtype=dtype,
    )

    kernels = []
    for s in sigma_values.tolist():
        ker = gaussian_kernel_2d_fixed_k(
            sigma=float(s),
            k=k,
            device=device,
            dtype=dtype,
        )  # [1,1,k,k]
        kernels.append(ker)

    kernels = torch.cat(kernels, dim=0)  # [S,1,k,k]
    return sigma_values, kernels


@torch.no_grad()
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
        x_bt1hw: [B, T, 1, H, W], values typically in [0,1]
        sigma_min: minimum sigma in palette
        sigma_max: maximum sigma in palette
        sigma_num: number of sigma samples in palette
        k: kernel size (odd)

    Returns:
        sigma_values: [S]
        y_btshw:      [B, T, S, H, W]

    Notes:
        - Output channel dimension S corresponds to blur palette index.
        - Because input channel is 1, we can flatten B and T, and use one conv2d
          with weight [S,1,k,k] to get all sigma outputs in parallel.
    """
    if x_bt1hw.ndim != 5:
        raise ValueError(f"x_bt1hw must be 5D [B,T,1,H,W], got shape={tuple(x_bt1hw.shape)}")

    B, T, C, H, W = x_bt1hw.shape
    if C != 1:
        raise ValueError(f"Expected input channel C=1, got C={C}")
    if k % 2 != 1:
        raise ValueError("k must be odd")

    device = x_bt1hw.device
    dtype = x_bt1hw.dtype

    sigma_values, kernels = build_sigma_palette(
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

    # One conv gets all sigma outputs at once:
    # input:  [BT, 1, H, W]
    # weight: [S, 1, k, k]
    # output: [BT, S, H, W]
    y_2d = F.conv2d(x_pad, kernels, bias=None, stride=1, padding=0, groups=1)

    y_btshw = y_2d.view(B, T, sigma_num, H, W)
    return sigma_values, y_btshw


@torch.no_grad()
def blur_bt1hw_from_strength_map(
    x_bt1hw: torch.Tensor,
    strength_bt1hw: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    sigma_num: int,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply spatially varying blur by interpolating between precomputed sigma palette outputs.

    Args:
        x_bt1hw:         [B, T, 1, H, W]
        strength_bt1hw:  [B, T, 1, H, W], values in [0,1]
        sigma_min: minimum sigma in palette
        sigma_max: maximum sigma in palette
        sigma_num: number of sigma samples in palette
        k: kernel size

    Returns:
        sigma_values: [S]
        palette_btshw: [B, T, S, H, W]
        y_bt1hw: [B, T, 1, H, W]

    Interpolation rule:
        strength=0   -> sigma_min
        strength=1   -> sigma_max

        Let u = strength * (S-1).
        Then interpolate between floor(u) and ceil(u).

    Example:
        if S=5, palette indices are [0,1,2,3,4]
        strength=0.4 -> u=1.6
        => interpolate 40% between idx=1 and idx=2? actually:
           low=1, high=2, alpha=0.6
           output = (1-alpha)*palette[1] + alpha*palette[2]
    """
    if x_bt1hw.ndim != 5:
        raise ValueError(f"x_bt1hw must be 5D [B,T,1,H,W], got shape={tuple(x_bt1hw.shape)}")
    if strength_bt1hw.ndim != 5:
        raise ValueError(
            f"strength_bt1hw must be 5D [B,T,1,H,W], got shape={tuple(strength_bt1hw.shape)}"
        )
    if x_bt1hw.shape != strength_bt1hw.shape:
        raise ValueError(
            f"x_bt1hw and strength_bt1hw must have same shape, "
            f"got {tuple(x_bt1hw.shape)} vs {tuple(strength_bt1hw.shape)}"
        )
    if sigma_num < 2:
        raise ValueError("sigma_num must be >= 2")

    B, T, C, H, W = x_bt1hw.shape
    if C != 1:
        raise ValueError(f"Expected input channel C=1, got C={C}")

    sigma_values, palette_btshw = blur_bt1hw_sigma_palette(
        x_bt1hw=x_bt1hw,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_num=sigma_num,
        k=k,
    )  # [B,T,S,H,W]

    strength = strength_bt1hw.clamp(0.0, 1.0)  # [B,T,1,H,W]
    u = strength * float(sigma_num - 1)        # [B,T,1,H,W]

    low_idx = torch.floor(u).long()            # [B,T,1,H,W]
    high_idx = torch.clamp(low_idx + 1, max=sigma_num - 1)
    alpha = (u - low_idx.to(u.dtype))          # [B,T,1,H,W]

    # Gather from palette along sigma dimension (dim=2)
    # palette_btshw: [B,T,S,H,W]
    # need index shape for gather: [B,T,1,H,W]
    low_val = torch.gather(palette_btshw, dim=2, index=low_idx)    # [B,T,1,H,W]
    high_val = torch.gather(palette_btshw, dim=2, index=high_idx)  # [B,T,1,H,W]

    y_bt1hw = (1.0 - alpha) * low_val + alpha * high_val
    return sigma_values, palette_btshw, y_bt1hw
