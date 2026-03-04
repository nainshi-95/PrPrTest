import torch

def brightness_mismatch_blocks(
    target: torch.Tensor,     # (B, T, H, W)
    reference: torch.Tensor,  # (B, T, R, H, W)
    block_size: int = 16,
    rho_threshold: float = 0.5,
    eps: float = 1e-8,
):
    """
    Detect brightness mismatch per 16x16 block using least squares fit X ≈ aZ + b.

    Returns:
        rho: (B, T, R, HB, WB)
        is_brightness_mismatch: (B, T, R, HB, WB) bool
        a: (B, T, R, HB, WB)
        b: (B, T, R, HB, WB)
    """

    B, T, H, W = target.shape
    _, _, R, _, _ = reference.shape

    BS = block_size
    HB = H // BS
    WB = W // BS

    # ---- block reshape
    tar = target.view(B, T, HB, BS, WB, BS).permute(0,1,2,4,3,5).contiguous()
    ref = reference.view(B, T, R, HB, BS, WB, BS).permute(0,1,2,3,5,4,6).contiguous()

    # shapes
    # tar: (B,T,HB,WB,BS,BS)
    # ref: (B,T,R,HB,WB,BS,BS)

    tar = tar.unsqueeze(2)  # (B,T,1,HB,WB,BS,BS)

    N = BS * BS

    # ---- compute means
    tar_mean = tar.mean(dim=(-2,-1), keepdim=True)
    ref_mean = ref.mean(dim=(-2,-1), keepdim=True)

    # ---- center
    tar_c = tar - tar_mean
    ref_c = ref - ref_mean

    # ---- compute a (scale)
    num = (tar_c * ref_c).sum(dim=(-2,-1))
    den = (ref_c ** 2).sum(dim=(-2,-1)) + eps

    a = num / den  # (B,T,R,HB,WB)

    # ---- compute b
    b = tar_mean.squeeze(-1).squeeze(-1) - a * ref_mean.squeeze(-1).squeeze(-1)

    # ---- reconstruct
    a_expand = a.unsqueeze(-1).unsqueeze(-1)
    b_expand = b.unsqueeze(-1).unsqueeze(-1)

    pred = a_expand * ref + b_expand

    # ---- energies
    E_before = ((tar - ref) ** 2).sum(dim=(-2,-1))
    E_after  = ((tar - pred) ** 2).sum(dim=(-2,-1))

    rho = E_after / (E_before + eps)

    # ---- brightness mismatch 판단
    is_brightness_mismatch = rho < rho_threshold

    return rho, is_brightness_mismatch, a, b
