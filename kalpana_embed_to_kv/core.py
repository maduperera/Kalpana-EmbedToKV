"""
Kalpanā Enterprise Memory Layer.
Powered exclusively by the compiled Kalpanā Native Kernel.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from kalpana.core import KalpanaRIFTensor as _NativeKernel
except ImportError:
    try:
        from kalpana.core import KalpanaEngineTensor as _NativeKernel
    except ImportError:
        _NativeKernel = None


class KalpanaRIFTensor(nn.Module):
    """
    Standard interface adapter for the Kalpanā Native Kernel.
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
        self.seen_tokens = 0

        if _NativeKernel is None:
            raise ImportError(
                "Kalpanā Native Kernel is required to initialize KalpanaRIFTensor. "
                "Please install kalpana_sdk_enterprise."
            )

        # Initialize native kernel without kwargs that native __init__ doesn't accept
        try:
            self._kernel = _NativeKernel(
                batch_size=batch_size,
                num_heads=num_heads,
                bands=bands,
                dim=dim,
                kappa=kappa,
                min_freq=min_freq,
                max_freq=max_freq,
                device=device,
            )
        except TypeError:
            self._kernel = _NativeKernel(
                batch_size=batch_size,
                num_heads=num_heads,
                bands=bands,
                dim=dim,
                device=device,
            )

    def write(self, t: Union[int, float, torch.Tensor], vector: torch.Tensor) -> None:
        """Writes token vector into native RIF kernel."""
        # Ensure vector shape is [batch, heads, seq_len, dim] or [batch, heads, dim]
        if vector.dim() == 3:
            # [batch, heads, dim] -> [batch, heads, 1, dim]
            vec_4d = vector.unsqueeze(2)
        elif vector.dim() == 2:
            # [batch, dim] -> [batch, num_heads, 1, dim]
            vec_4d = vector.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, -1, -1)
        else:
            vec_4d = vector

        start_t = int(t) if not isinstance(t, torch.Tensor) else int(t.item())
        
        if hasattr(self._kernel, "write_rif"):
            self._kernel.write_rif(start_t, vec_4d.to(self.device, dtype=torch.float32))
        elif hasattr(self._kernel, "write"):
            self._kernel.write(start_t, vec_4d)
        
        self.seen_tokens += vec_4d.shape[2]

    def batch_reconstruct(self, t_range: torch.Tensor) -> torch.Tensor:
        """Reconstructs past key/value vectors from native RIF kernel."""
        max_t = len(t_range)
        if hasattr(self._kernel, "reconstruct_all"):
            out = self._kernel.reconstruct_all(max_t)
            return out.to(self.device, dtype=self.dtype)
        elif hasattr(self._kernel, "batch_reconstruct"):
            return self._kernel.batch_reconstruct(t_range).to(self.device, dtype=self.dtype)
        elif hasattr(self._kernel, "reconstruct"):
            # reconstruct per step
            recons = [self._kernel.reconstruct(i) for i in range(max_t)]
            return torch.stack(recons, dim=2).to(self.device, dtype=self.dtype)
        raise AttributeError("Native kernel has no reconstruction method")

    def memory_footprint_mb(self) -> float:
        """Calculates constant O(1) memory footprint."""
        if hasattr(self._kernel, "memory_footprint_mb"):
            return self._kernel.memory_footprint_mb()
        # Fallback calculation based on tensor dimensions: real + imag buffers
        total_elements = self.batch_size * self.num_heads * self.bands * self.dim * 2
        element_size = 4 # float32
        return (total_elements * element_size) / (1024 * 1024)

    def reset(self) -> None:
        if hasattr(self._kernel, "reset"):
            self._kernel.reset()
        self.seen_tokens = 0


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
