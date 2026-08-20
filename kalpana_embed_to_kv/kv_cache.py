"""
KV Cache Replacement for Transformers and HuggingFace LLM generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from .core import KalpanaRIFTensor


class KalpanaKVCache:
    """
    Multi-layer O(1) Key-Value Cache manager for Auto-Regressive Language Models.
    Maintains independent fixed-size RIF tensors per transformer layer.
    """
    def __init__(
        self,
        num_layers: int,
        batch_size: int = 1,
        num_heads: int = 32,
        head_dim: int = 128,
        bands: int = 2048,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bands = bands
        self.device = device
        self.dtype = dtype

        self.key_layers: List[KalpanaRIFTensor] = []
        self.value_layers: List[KalpanaRIFTensor] = []

        for _ in range(num_layers):
            self.key_layers.append(
                KalpanaRIFTensor(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    bands=bands,
                    dim=head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.value_layers.append(
                KalpanaRIFTensor(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    bands=bands,
                    dim=head_dim,
                    device=device,
                    dtype=dtype,
                )
            )

        self._seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new incoming Key and Value tensors.
        
        Args:
            key_states: Tensor [batch_size, num_heads, seq_len_new, head_dim]
            value_states: Tensor [batch_size, num_heads, seq_len_new, head_dim]
            layer_idx: Layer index to update
            cache_kwargs: Extra options
            
        Returns:
            Tuple of (all_keys, all_values) spanning the entire context window.
        """
        seq_len_new = key_states.shape[2]
        
        # Inject each incoming token into RIF
        for step in range(seq_len_new):
            t = self._seen_tokens + step
            k_t = key_states[:, :, step, :]
            v_t = value_states[:, :, step, :]
            self.key_layers[layer_idx].write(t, k_t)
            self.value_layers[layer_idx].write(t, v_t)

        total_tokens = self._seen_tokens + seq_len_new
        t_range = torch.arange(0, total_tokens, device=self.device).float()

        past_k = self.key_layers[layer_idx].batch_reconstruct(t_range)
        past_v = self.value_layers[layer_idx].batch_reconstruct(t_range)

        if layer_idx == self.num_layers - 1:
            self._seen_tokens = total_tokens

        return past_k, past_v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the number of tokens stored in the cache."""
        return self._seen_tokens

    def get_total_memory_mb(self) -> float:
        """Computes combined memory of all layers in Megabytes (MB). Constant O(1)."""
        total = 0.0
        for k_rif, v_rif in zip(self.key_layers, self.value_layers):
            total += k_rif.memory_footprint_mb() + v_rif.memory_footprint_mb()
        return total

    def reset(self) -> None:
        """Clears all layer states."""
        for k_rif, v_rif in zip(self.key_layers, self.value_layers):
            k_rif.reset()
            v_rif.reset()
        self._seen_tokens = 0


class KalpanaDynamicCache:
    """
    HuggingFace compatible DynamicCache adapter interface.
    Allows seamless drop-in into model.generate(..., past_key_values=kalpana_cache).
    """
    def __init__(
        self,
        num_layers: int = 32,
        batch_size: int = 1,
        num_heads: int = 32,
        head_dim: int = 128,
        bands: int = 2048,
        device: Union[str, torch.device] = 'cpu',
    ):
        self.engine = KalpanaKVCache(
            num_layers=num_layers,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bands=bands,
            device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.engine.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.engine.get_seq_length(layer_idx)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.engine.get_seq_length(layer_idx)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t_range = torch.arange(0, self.engine.get_seq_length(layer_idx), device=self.engine.device).float()
        k = self.engine.key_layers[layer_idx].batch_reconstruct(t_range)
        v = self.engine.value_layers[layer_idx].batch_reconstruct(t_range)
        return k, v

    def __len__(self) -> int:
        return self.engine.num_layers
