"""
Kalpanā O(1) Memory Substrate Integration.
Standard PyTorch tensor interface with binary kernel acceleration.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from kalpana.core import KalpanaEngineTensor as _NativeRIF
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class KalpanaRIFTensor(nn.Module):
    """
    Kalpanā O(1) Memory Substrate.
    Maintains a constant-size state tensor regardless of sequence length.
    """
    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        bands: int = 512,
        dim: int = 128,
        kappa: float = 1.0,
        min_freq: float = 0.1,
        max_freq: float = 10.0,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.bands = bands
        self.dim = dim
        self.kappa = kappa
        self.device = device
        self.dtype = dtype

        # State storage buffers
        self.register_buffer(
            "state_re",
            torch.zeros(batch_size, num_heads, bands, dim, device=device, dtype=dtype)
        )
        self.register_buffer(
            "state_im",
            torch.zeros(batch_size, num_heads, bands, dim, device=device, dtype=dtype)
        )

        # Basis frequency & phase parameter buffers
        b_val = float(bands - 1) if bands > 1 else 1.0
        delta = (max_freq - min_freq) / b_val
        w_basis = min_freq + torch.arange(bands, device=device, dtype=dtype) * delta
        self.register_buffer("_w", w_basis.view(1, 1, bands, 1))

        phi_basis = torch.linspace(0, 6.28318530718, bands, device=device, dtype=dtype)
        self.register_buffer("_phi", phi_basis.view(1, 1, bands, 1))

        self.total_tokens_stored = 0

    def write(self, t: Union[int, float, torch.Tensor], vector: torch.Tensor) -> None:
        """Projects vector into the continuous state substrate at coordinate t."""
        if vector.dim() == 2:
            vector = vector.unsqueeze(1).expand(-1, self.num_heads, -1)

        v_exp = vector.unsqueeze(2)

        if not isinstance(t, torch.Tensor):
            t_val = float(t)
        else:
            t_val = t.to(self.device, dtype=self.dtype)

        theta = self.kappa * self._w * t_val + self._phi
        self.state_re.add_(v_exp * torch.cos(theta))
        self.state_im.add_(v_exp * torch.sin(theta))
        self.total_tokens_stored += 1

    def reconstruct(self, t: Union[int, float, torch.Tensor]) -> torch.Tensor:
        """Reconstructs state at coordinate t."""
        if not isinstance(t, torch.Tensor):
            t_val = float(t)
        else:
            t_val = t.to(self.device, dtype=self.dtype)

        theta = self.kappa * self._w * t_val + self._phi
        rv = self.state_re * torch.cos(theta) + self.state_im * torch.sin(theta)
        return rv.mean(dim=2)

    def batch_reconstruct(self, t_range: torch.Tensor) -> torch.Tensor:
        """Batched tensor contraction reconstruction."""
        t_seq = t_range.to(self.device, dtype=self.dtype)
        theta_mat = torch.outer(t_seq, (self.kappa * self._w).view(-1)) + self._phi.view(-1).unsqueeze(0)
        
        re_part = torch.einsum('tk,nhkd->nhtd', torch.cos(theta_mat), self.state_re)
        im_part = torch.einsum('tk,nhkd->nhtd', torch.sin(theta_mat), self.state_im)

        return (re_part + im_part) / float(self.bands)

    def memory_footprint_mb(self) -> float:
        re_mb = self.state_re.nelement() * self.state_re.element_size() / (1024 * 1024)
        im_mb = self.state_im.nelement() * self.state_im.element_size() / (1024 * 1024)
        return re_mb + im_mb

    def reset(self) -> None:
        self.state_re.zero_()
        self.state_im.zero_()
        self.total_tokens_stored = 0


class EmbedToKVMatrix(nn.Module):
    """Semantic embedding to KV transformation adapter."""
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = embeddings.shape
        k = self.k_proj(embeddings).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(embeddings).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v
