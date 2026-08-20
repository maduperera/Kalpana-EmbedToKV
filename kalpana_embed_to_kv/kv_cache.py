"""
KV Cache Replacement for Transformers and HuggingFace LLM generation.
Compatible with transformers.cache_utils.Cache and CacheLayerMixin.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from .core import KalpanaRIFTensor

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
    _HAS_TRANSFORMERS_CACHE = True
except ImportError:
    class Cache:
        def __init__(self, *args, **kwargs):
            pass
    class CacheLayerMixin:
        pass
    _HAS_TRANSFORMERS_CACHE = False


class KalpanaCacheLayer(CacheLayerMixin):
    """
    Individual Transformer Layer Cache powered by O(1) RIF Holographic Memory.
    Replaces torch.cat with constant-memory wave interference superposition.
    """
    is_sliding = False

    def __init__(self, bands: int = 4096, kappa: float = 1.0):
        if _HAS_TRANSFORMERS_CACHE:
            super().__init__()
        self.bands = bands
        self.kappa = kappa
        self.key_rif: Optional[KalpanaRIFTensor] = None
        self.val_rif: Optional[KalpanaRIFTensor] = None
        self.seen_tokens = 0
        self.is_initialized = False
        self.device = "cpu"
        self.dtype = torch.float32

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        batch_size, num_heads, _, head_dim = key_states.shape
        self.device = key_states.device
        self.dtype = key_states.dtype

        self.key_rif = KalpanaRIFTensor(
            batch_size=batch_size,
            num_heads=num_heads,
            bands=self.bands,
            dim=head_dim,
            kappa=self.kappa,
            device=self.device,
            dtype=self.dtype,
        )
        self.val_rif = KalpanaRIFTensor(
            batch_size=batch_size,
            num_heads=num_heads,
            bands=self.bands,
            dim=head_dim,
            kappa=self.kappa,
            device=self.device,
            dtype=self.dtype,
        )
        self.seen_tokens = 0
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size, num_heads, seq_len_new, head_dim = key_states.shape

        # Inject each incoming token into RIF O(1) state
        for step in range(seq_len_new):
            t = self.seen_tokens + step
            k_t = key_states[:, :, step, :]
            v_t = value_states[:, :, step, :]
            self.key_rif.write(t, k_t)
            self.val_rif.write(t, v_t)

        self.seen_tokens += seq_len_new
        t_range = torch.arange(0, self.seen_tokens, device=self.device).float()

        reconstructed_keys = self.key_rif.batch_reconstruct(t_range)
        reconstructed_values = self.val_rif.batch_reconstruct(t_range)

        return reconstructed_keys, reconstructed_values

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.seen_tokens + query_length, 0

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def get_max_cache_shape(self) -> int:
        return -1

    def memory_footprint_mb(self) -> float:
        if not self.is_initialized or self.key_rif is None:
            return 0.0
        return self.key_rif.memory_footprint_mb() + self.val_rif.memory_footprint_mb()

    def reset(self) -> None:
        if self.is_initialized and self.key_rif is not None:
            self.key_rif.reset()
            self.val_rif.reset()
        self.seen_tokens = 0


class KalpanaDynamicCache(Cache):
    """
    Drop-in O(1) Memory KV Cache replacement for HuggingFace Transformers.
    """
    def __init__(
        self,
        num_layers: Optional[int] = None,
        bands: int = 4096,
        kappa: float = 1.0,
        **kwargs,
    ):
        self.bands = bands
        self.kappa = kappa
        
        if _HAS_TRANSFORMERS_CACHE:
            if num_layers is not None:
                layers = [KalpanaCacheLayer(bands=bands, kappa=kappa) for _ in range(num_layers)]
                super().__init__(layers=layers)
            else:
                # Custom factory for dynamic instantiation per layer
                class LayerFactory(KalpanaCacheLayer):
                    def __init__(self, *args, **kw):
                        super().__init__(bands=bands, kappa=kappa)
                super().__init__(layer_class_to_replicate=LayerFactory)
        else:
            self.layers = [KalpanaCacheLayer(bands=bands, kappa=kappa) for _ in range(num_layers or 32)]

    def get_total_memory_mb(self) -> float:
        total = 0.0
        for layer in self.layers:
            if isinstance(layer, KalpanaCacheLayer):
                total += layer.memory_footprint_mb()
        return total

# Alias for backwards compatibility
KalpanaKVCache = KalpanaDynamicCache
