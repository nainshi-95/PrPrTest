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
