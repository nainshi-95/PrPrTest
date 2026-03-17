import math
from typing import Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDCTQuantizer(nn.Module):
    """
    Block-wise DCT -> QP-based quantization -> IDCT module.

    Notes
    -----
    - This is a simplified VVC/HEVC-style QP-to-qstep approximation:
          qstep = 2 ** ((qp - 4) / 6)
      not the exact normative VVC quantization.
    - DCT/IDCT are implemented with orthonormal DCT-II matrices.
    - DCT matrices are cached by (block_size, device, dtype).
    - Qstep tensors are cached by (qp, device, dtype).

    Supported input shapes
    ----------------------
    - [H, W]
    - [C, H, W]
    - [B, C, H, W]

    Output shape matches input shape.
    """

    def __init__(self, pad_mode: str = "replicate"):
        super().__init__()
        self.pad_mode = pad_mode

        # Runtime caches (not persistent in state_dict by default)
        self._dct_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._idct_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._qstep_cache: Dict[Tuple[float, str, str], torch.Tensor] = {}

    @staticmethod
    def _dtype_key(dtype: torch.dtype) -> str:
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.type == "cuda":
            return f"cuda:{device.index}"
        return device.type

    def _make_dct_matrix(
        self,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create orthonormal DCT-II matrix of size [N, N].
        """
        n = torch.arange(N, device=device, dtype=dtype)
        k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)

        # DCT-II basis
        D = torch.cos(math.pi / N * (n + 0.5) * k)

        # Orthonormal scaling
        D[0, :] *= math.sqrt(1.0 / N)
        if N > 1:
            D[1:, :] *= math.sqrt(2.0 / N)

        return D

    def _get_dct_idct(
        self,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached DCT and IDCT matrices.
        For orthonormal DCT-II, IDCT is simply D^T.
        """
        key = (N, self._device_key(device), self._dtype_key(dtype))

        if key not in self._dct_cache:
            D = self._make_dct_matrix(N, device=device, dtype=dtype)
            ID = D.transpose(-1, -2).contiguous()

            self._dct_cache[key] = D
            self._idct_cache[key] = ID

        return self._dct_cache[key], self._idct_cache[key]

    def _get_qstep(
        self,
        qp: Union[int, float, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Simplified VVC/HEVC-style QP to quantization step:
            qstep = 2 ^ ((qp - 4) / 6)

        Returns a scalar tensor on the target device/dtype.
        """
        if isinstance(qp, torch.Tensor):
            if qp.numel() != 1:
                raise ValueError("qp tensor must be scalar.")
            qp_val = float(qp.detach().cpu().item())
        else:
            qp_val = float(qp)

        key = (qp_val, self._device_key(device), self._dtype_key(dtype))

        if key not in self._qstep_cache:
            qstep = 2.0 ** ((qp_val - 4.0) / 6.0)
            self._qstep_cache[key] = torch.tensor(qstep, device=device, dtype=dtype)

        return self._qstep_cache[key]

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert input to BCHW and return original ndim for restoration.
        """
        if x.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            return x.unsqueeze(0).unsqueeze(0), 2
        elif x.ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            return x.unsqueeze(0), 3
        elif x.ndim == 4:
            # [B, C, H, W]
            return x, 4
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    @staticmethod
    def _from_bchw(x: torch.Tensor, original_ndim: int) -> torch.Tensor:
        """
        Restore original shape from BCHW.
        """
        if original_ndim == 2:
            return x[0, 0]
        elif original_ndim == 3:
            return x[0]
        elif original_ndim == 4:
            return x
        else:
            raise ValueError(f"Unsupported original ndim: {original_ndim}")

    @staticmethod
    def _blocks_to_image(blocks: torch.Tensor) -> torch.Tensor:
        """
        Convert block tensor [B, C, nH, nW, N, N] back to image [B, C, H, W].
        """
        B, C, nH, nW, N, _ = blocks.shape
        x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, nH * N, nW * N)
        return x

    def forward(
        self,
        x: torch.Tensor,
        block_size: int,
        qp: Union[int, float, torch.Tensor],
        return_intermediates: bool = False,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor with shape [H,W], [C,H,W], or [B,C,H,W]
        block_size : int
            Block size N
        qp : int | float | torch.Tensor(scalar)
            Quantization parameter
        return_intermediates : bool
            If True, also returns coeff, qcoeff, dequant_coeff

        Returns
        -------
        recon : torch.Tensor
            Reconstructed tensor with same shape as input
        optionally:
            coeff, qcoeff, dequant_coeff
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        x_bchw, original_ndim = self._to_bchw(x)
        B, C, H, W = x_bchw.shape
        N = block_size

        device = x_bchw.device
        dtype = x_bchw.dtype

        # Pad so that H and W are multiples of N
        pad_h = (N - (H % N)) % N
        pad_w = (N - (W % N)) % N

        if pad_h > 0 or pad_w > 0:
            # F.pad order: (left, right, top, bottom)
            x_pad = F.pad(x_bchw, (0, pad_w, 0, pad_h), mode=self.pad_mode)
        else:
            x_pad = x_bchw

        _, _, Hpad, Wpad = x_pad.shape
        nH = Hpad // N
        nW = Wpad // N

        # [B, C, Hpad, Wpad] -> [B, C, nH, nW, N, N]
        blocks = x_pad.unfold(2, N, N).unfold(3, N, N).contiguous()

        D, ID = self._get_dct_idct(N, device=device, dtype=dtype)
        qstep = self._get_qstep(qp, device=device, dtype=dtype)

        # Forward DCT: coeff = D @ block @ D^T
        # blocks shape: [B, C, nH, nW, N, N]
        coeff = torch.matmul(D, blocks)
        coeff = torch.matmul(coeff, D.transpose(-1, -2))

        # Quantize / dequantize
        qcoeff = torch.round(coeff / qstep)
        dequant_coeff = qcoeff * qstep

        # IDCT: recon_block = D^T @ dequant_coeff @ D
        recon_blocks = torch.matmul(ID, dequant_coeff)
        recon_blocks = torch.matmul(recon_blocks, ID.transpose(-1, -2))

        recon_pad = self._blocks_to_image(recon_blocks)

        # Remove padding
        recon = recon_pad[:, :, :H, :W]
        recon = self._from_bchw(recon, original_ndim)

        if return_intermediates:
            return recon, coeff, qcoeff, dequant_coeff
        return recon
