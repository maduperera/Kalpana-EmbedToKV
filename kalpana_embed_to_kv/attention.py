"""
Kalpana Attention Layers: Native transformer attention execution with O(1) RIF KV cache.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from .core import KalpanaRIFTensor


class KalpanaAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer equipped with internal O(1) RIF Key-Value memory.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        bands: int = 2048,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.key_rif = KalpanaRIFTensor(
            batch_size=1, num_heads=num_heads, bands=bands, dim=self.head_dim, device=device, dtype=dtype
        )
        self.val_rif = KalpanaRIFTensor(
            batch_size=1, num_heads=num_heads, bands=bands, dim=self.head_dim, device=device, dtype=dtype
        )

        self.current_t = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for auto-regressive decoding or prompt ingestion.
        
        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            use_cache: Whether to update & read RIF KV memory
            
        Returns:
            Tuple of (output_states, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.key_rif.batch_size != batch_size:
            self.key_rif = KalpanaRIFTensor(
                batch_size=batch_size, num_heads=self.num_heads,
                bands=self.key_rif.bands, dim=self.head_dim, device=self.device, dtype=self.dtype
            )
            self.val_rif = KalpanaRIFTensor(
                batch_size=batch_size, num_heads=self.num_heads,
                bands=self.val_rif.bands, dim=self.head_dim, device=self.device, dtype=self.dtype
            )

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Ingest new tokens into RIF
        for step in range(seq_len):
            t = self.current_t + step
            self.key_rif.write(t, k[:, :, step, :])
            self.val_rif.write(t, v[:, :, step, :])

        total_t = self.current_t + seq_len
        t_range = torch.arange(0, total_t, device=self.device).float()

        # Reconstruct all past Keys & Values from holographic fields
        all_keys = self.key_rif.batch_reconstruct(t_range)    # [batch, heads, total_t, head_dim]
        all_values = self.val_rif.batch_reconstruct(t_range)  # [batch, heads, total_t, head_dim]

        # Scaled Dot-Product Attention: [batch, heads, seq_len, total_t]
        scores = torch.matmul(q, all_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, all_values) # [batch, heads, seq_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        self.current_t = total_t
        return output, attn_weights

    def reset_cache(self) -> None:
        """Resets the KV memory states."""
        self.key_rif.reset()
        self.val_rif.reset()
        self.current_t = 0


class KalpanaKVInterpreter(nn.Module):
    """
    Direct interpreter wrapping Key/Value injection and attention score resolution.
    """
    def __init__(self, batch_size: int, num_heads: int, bands: int, dim: int, device: str = 'cpu'):
        super().__init__()
        self.key_rif = KalpanaRIFTensor(batch_size, num_heads, bands, dim, device=device)
        self.val_rif = KalpanaRIFTensor(batch_size, num_heads, bands, dim, device=device)
        self.dim = dim
        self.current_t = 0
        self.device = device

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates attention output for a single step (token) against holographic history.
        q, k, v: [batch_size, num_heads, dim]
        """
        self.key_rif.write(self.current_t, k)
        self.val_rif.write(self.current_t, v)

        t_range = torch.arange(0, self.current_t + 1, device=self.device).float()
        past_keys = self.key_rif.batch_reconstruct(t_range)
        past_values = self.val_rif.batch_reconstruct(t_range)

        q_unsqueezed = q.unsqueeze(2) # [batch, heads, 1, dim]
        scores = torch.matmul(q_unsqueezed, past_keys.transpose(-2, -1)) / math.sqrt(self.dim)
        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, past_values)
        self.current_t += 1

        return out.squeeze(2), attn_weights
