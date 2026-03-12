_HADAMARD_CACHE = {}



import torch

def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    n must be power of 2.
    Returns (n, n) Hadamard matrix.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")

    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H


def satd_2d(residual: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    residual: (..., H, W), where H and W are powers of 2
    returns: (...,) SATD

    SATD = sum(abs(H * residual * H^T))
    """
    if residual.ndim < 2:
        raise ValueError("residual must have at least 2 dims")

    h, w = residual.shape[-2], residual.shape[-1]
    if (h & (h - 1)) != 0 or (w & (w - 1)) != 0:
        raise ValueError("H and W must be powers of 2")

    device = residual.device
    dtype = residual.dtype

    Hh = hadamard_matrix(h, device=device, dtype=dtype)
    Hw = hadamard_matrix(w, device=device, dtype=dtype)

    # (..., H, W)
    transformed = Hh @ residual @ Hw.transpose(-1, -2)
    satd = transformed.abs().sum(dim=(-2, -1))

    if normalize:
        # Many codecs divide by sqrt(H*W) or use block-specific shifts.
        # Here we use a simple normalization.
        satd = satd / (h * w) ** 0.5

    return satd
    



def get_hadamard(n: int, device, dtype):
    key = (n, device.type, device.index, dtype)
    if key not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[key] = hadamard_matrix(n, device=device, dtype=dtype)
    return _HADAMARD_CACHE[key]


def satd_2d_fast(residual: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    h, w = residual.shape[-2], residual.shape[-1]
    Hh = get_hadamard(h, residual.device, residual.dtype)
    Hw = get_hadamard(w, residual.device, residual.dtype)

    transformed = Hh @ residual @ Hw.transpose(-1, -2)
    satd = transformed.abs().sum(dim=(-2, -1))

    if normalize:
        satd = satd / (h * w) ** 0.5
    return satd







def block_satd(org: torch.Tensor, pred: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    org, pred: (..., H, W)
    """
    if org.shape != pred.shape:
        raise ValueError("org and pred must have same shape")
    residual = org - pred
    return satd_2d(residual, normalize=normalize)










def frame_to_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    """
    x: (B, 1, H, W)
    returns: (B, nH, nW, block, block)
    """
    B, C, H, W = x.shape
    if C != 1:
        raise ValueError("Only single-channel input supported here")
    if H % block != 0 or W % block != 0:
        raise ValueError("H and W must be divisible by block size")

    patches = x.unfold(2, block, block).unfold(3, block, block)
    # (B,1,nH,nW,block,block)
    patches = patches.squeeze(1)
    return patches


def blockwise_satd(org: torch.Tensor, pred: torch.Tensor, block: int = 8) -> torch.Tensor:
    """
    org, pred: (B, 1, H, W)
    returns: (B, nH, nW)
    """
    org_blk = frame_to_blocks(org, block)
    pred_blk = frame_to_blocks(pred, block)
    diff = org_blk - pred_blk
    return satd_2d(diff, normalize=False)






