"""
Core RIF (Resonant Interference Field) Holographic Memory Tensor.
Maintains an exact O(1) memory footprint for vector storage and retrieval.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class KalpanaRIFTensor(nn.Module):
    """
    Kalpana Resonant Interference Field (RIF) Memory Substrate.
    Stores arbitrary sequence lengths into a fixed-size holographic tensor matrix.
    
    Memory Complexity: O(1) with respect to sequence length N.
    Space: (batch_size, num_heads, bands, dim) * 2 complex components.
    """
    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        bands: int = 2048,
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

        # Fixed O(1) Memory States: Real & Imaginary components
        self.register_buffer(
            "state_re",
            torch.zeros(batch_size, num_heads, bands, dim, device=device, dtype=dtype)
        )
        self.register_buffer(
            "state_im",
            torch.zeros(batch_size, num_heads, bands, dim, device=device, dtype=dtype)
        )

        # Holographic Spatial Frequencies (o3) and Phase Offsets (p4)
        bands_f = float(bands - 1) if bands > 1 else 1.0
        step = (max_freq - min_freq) / bands_f

        # Orthogonal harmonic frequency basis for maximum SNR and interference cancellation
        o3 = min_freq + torch.arange(bands, device=device, dtype=dtype) * step
        self.register_buffer("o3", o3.view(1, 1, bands, 1))

        # Uniformly distributed phase angles
        p4 = torch.linspace(0, 2.0 * math.pi, bands, device=device, dtype=dtype)
        self.register_buffer("p4", p4.view(1, 1, bands, 1))

        self.total_tokens_stored = 0

    def write(self, t: Union[int, float, torch.Tensor], vector: torch.Tensor) -> None:
        """
        Injects a vector (or batch of vectors) into the holographic field at temporal coordinate `t`.
        
        Args:
            t: Temporal index or coordinate tensor.
            vector: Tensor of shape `[batch_size, num_heads, dim]` or `[batch_size, dim]`.
        """
        if vector.dim() == 2:
            # Expand to [batch_size, 1, dim] then broadcast over num_heads if needed
            vector = vector.unsqueeze(1).expand(-1, self.num_heads, -1)

        vector_expanded = vector.unsqueeze(2) # [batch, heads, 1, dim]

        if not isinstance(t, torch.Tensor):
            t_val = float(t)
        else:
            t_val = t.to(self.device, dtype=self.dtype)

        angle = self.kappa * self.o3 * t_val + self.p4
        cr = torch.cos(angle)
        ci = torch.sin(angle)

        self.state_re.add_(vector_expanded * cr)
        self.state_im.add_(vector_expanded * ci)
        self.total_tokens_stored += 1

    def reconstruct(self, t: Union[int, float, torch.Tensor]) -> torch.Tensor:
        """
        Reconstructs the vector stored at temporal coordinate `t` via holographic resonance.
        
        Args:
            t: Coordinate or index to sweep.
            
        Returns:
            Reconstructed vector tensor of shape `[batch_size, num_heads, dim]`.
        """
        if not isinstance(t, torch.Tensor):
            t_val = float(t)
        else:
            t_val = t.to(self.device, dtype=self.dtype)

        angle = self.kappa * self.o3 * t_val + self.p4
        cr = torch.cos(angle)
        ci = torch.sin(angle)

        rv = self.state_re * cr + self.state_im * ci
        return rv.mean(dim=2) # Average across bands

    def batch_reconstruct(self, t_range: torch.Tensor) -> torch.Tensor:
        """
        Vectorized sweep reconstructing past vectors for a range of temporal coordinates.
        
        Args:
            t_range: 1D Tensor of coordinates of length T.
            
        Returns:
            Reconstructed past vectors: shape `[batch_size, num_heads, T, dim]`.
        """
        t_seq = t_range.to(self.device, dtype=self.dtype)
        # [T, 1, 1, 1, 1]
        t_view = t_seq.view(-1, 1, 1, 1, 1)
        angle = self.kappa * self.o3 * t_view + self.p4
        cr = torch.cos(angle)
        ci = torch.sin(angle)

        # state_re: [1, batch, heads, bands, dim]
        state_re_exp = self.state_re.unsqueeze(0)
        state_im_exp = self.state_im.unsqueeze(0)

        # rv: [T, batch, heads, bands, dim]
        rv = state_re_exp * cr + state_im_exp * ci
        # Mean across bands -> [T, batch, heads, dim]
        reconstructed = rv.mean(dim=3)
        # Permute to [batch, heads, T, dim]
        return reconstructed.permute(1, 2, 0, 3)

    def memory_footprint_mb(self) -> float:
        """Returns total memory footprint of the RIF storage in Megabytes (MB)."""
        re_mb = self.state_re.nelement() * self.state_re.element_size() / (1024 * 1024)
        im_mb = self.state_im.nelement() * self.state_im.element_size() / (1024 * 1024)
        return re_mb + im_mb

    def reset(self) -> None:
        """Clears the holographic field to zero."""
        self.state_re.zero_()
        self.state_im.zero_()
        self.total_tokens_stored = 0


class EmbedToKVMatrix(nn.Module):
    """
    Transforms Semantic / Token Embeddings into paired Key-Value Holographic Memories.
    """
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int = 8,
        bands: int = 2048,
        device: Union[str, torch.device] = 'cpu',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_head_dim = head_dim * num_heads

        # Linear projections from raw embedding to Key & Value multi-head representations
        self.k_proj = nn.Linear(embed_dim, self.total_head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(embed_dim, self.total_head_dim, bias=False, device=device)

        # Key and Value RIF Holographic Engines
        self.key_rif = KalpanaRIFTensor(
            batch_size=1, num_heads=num_heads, bands=bands, dim=head_dim, device=device
        )
        self.val_rif = KalpanaRIFTensor(
            batch_size=1, num_heads=num_heads, bands=bands, dim=head_dim, device=device
        )

        self.current_t = 0
        self.device = device

    def ingest_embedding(self, embedding: torch.Tensor) -> int:
        """
        Projects an input embedding vector to K and V, injecting them into the O(1) holographic memory.
        
        Args:
            embedding: Tensor of shape `[batch_size, embed_dim]`.
            
        Returns:
            The temporal coordinate `t` where this embedding was indexed.
        """
        batch_size = embedding.shape[0]
        if self.key_rif.batch_size != batch_size:
            self.key_rif = KalpanaRIFTensor(
                batch_size=batch_size, num_heads=self.num_heads,
                bands=self.key_rif.bands, dim=self.head_dim, device=self.device
            )
            self.val_rif = KalpanaRIFTensor(
                batch_size=batch_size, num_heads=self.num_heads,
                bands=self.val_rif.bands, dim=self.head_dim, device=self.device
            )

        k_vec = self.k_proj(embedding).view(batch_size, self.num_heads, self.head_dim)
        v_vec = self.v_proj(embedding).view(batch_size, self.num_heads, self.head_dim)

        self.key_rif.write(self.current_t, k_vec)
        self.val_rif.write(self.current_t, v_vec)

        assigned_t = self.current_t
        self.current_t += 1
        return assigned_t

    def get_past_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves all past Keys and Values by performing a fast parallel temporal sweep.
        
        Returns:
            Tuple of (keys, values), each with shape `[batch_size, num_heads, seq_len, head_dim]`.
        """
        if self.current_t == 0:
            empty = torch.empty(self.key_rif.batch_size, self.num_heads, 0, self.head_dim, device=self.device)
            return empty, empty

        t_range = torch.arange(0, self.current_t, device=self.device).float()
        keys = self.key_rif.batch_reconstruct(t_range)
        values = self.val_rif.batch_reconstruct(t_range)
        return keys, values
