_HADAMARD_CACHE = {}

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






