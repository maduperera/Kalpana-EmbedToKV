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
class KalpanaDynamicCache(Cache):
    """
    Drop-in pure O(1) Memory KV Cache replacement for HuggingFace Transformers.
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
                class LayerFactory(KalpanaCacheLayer):
                    def __init__(self, *args, **kw):
                        super().__init__(bands=bands, kappa=kappa)
                super().__init__(layer_class_to_replicate=LayerFactory)
        else:
            self.layers = [KalpanaCacheLayer(bands=bands, kappa=kappa) for _ in range(num_layers or 32)]

    def get_total_memory_mb(self) -> float:
        total = 0.0
        for layer in self.layers:
            if hasattr(layer, "memory_footprint_mb"):
                total += layer.memory_footprint_mb()
        return total


class KalpanaHybridCacheLayer(CacheLayerMixin):
    """
    Hybrid Cache Layer:
    - Maintains exact high-precision KV states for the recent local window (e.g. 128 tokens).
    - Stores all long-range tokens (up to 1M+) in the fixed O(1) RIF holographic memory.
    """
    is_sliding = False

    def __init__(self, sliding_window: int = 128, bands: int = 4096, kappa: float = 1.0):
        if _HAS_TRANSFORMERS_CACHE:
            super().__init__()
        self.sliding_window = sliding_window
        self.bands = bands
        self.kappa = kappa
        self.key_rif: Optional[KalpanaRIFTensor] = None
        self.val_rif: Optional[KalpanaRIFTensor] = None
        self.local_keys: Optional[torch.Tensor] = None
        self.local_values: Optional[torch.Tensor] = None
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
        self.local_keys = torch.empty(batch_size, num_heads, 0, head_dim, device=self.device, dtype=self.dtype)
        self.local_values = torch.empty(batch_size, num_heads, 0, head_dim, device=self.device, dtype=self.dtype)
        self.seen_tokens = 0
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        batch_size, num_heads, seq_len_new, head_dim = key_states.shape

        # 1. Ingest every token into RIF O(1) state
        for step in range(seq_len_new):
            t = self.seen_tokens + step
            k_t = key_states[:, :, step, :]
            v_t = value_states[:, :, step, :]
            self.key_rif.write(t, k_t)
            self.val_rif.write(t, v_t)

        self.seen_tokens += seq_len_new

        # 2. Append to local sliding buffer
        self.local_keys = torch.cat([self.local_keys, key_states], dim=2)
        self.local_values = torch.cat([self.local_values, value_states], dim=2)

        # 3. If within local window, return exact local buffer
        if self.seen_tokens <= self.sliding_window:
            return self.local_keys, self.local_values

        # Keep only the last sliding_window tokens in the local buffer
        if self.local_keys.shape[2] > self.sliding_window:
            self.local_keys = self.local_keys[:, :, -self.sliding_window:, :]
            self.local_values = self.local_values[:, :, -self.sliding_window:, :]

        # 4. Reconstruct long-range prefix from RIF
        long_range_len = self.seen_tokens - self.sliding_window
        t_range_prefix = torch.arange(0, long_range_len, device=self.device).float()
        prefix_keys = self.key_rif.batch_reconstruct(t_range_prefix)
        prefix_values = self.val_rif.batch_reconstruct(t_range_prefix)

        # 5. Concatenate long-range RIF prefix + exact sliding window
        full_keys = torch.cat([prefix_keys, self.local_keys], dim=2)
        full_values = torch.cat([prefix_values, self.local_values], dim=2)

        return full_keys, full_values

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.seen_tokens + query_length, 0

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def get_max_cache_shape(self) -> int:
        return -1

    def memory_footprint_mb(self) -> float:
        if not self.is_initialized or self.key_rif is None:
            return 0.0
        rif_mb = self.key_rif.memory_footprint_mb() + self.val_rif.memory_footprint_mb()
        local_mb = (self.local_keys.nelement() * self.local_keys.element_size() * 2) / (1024 * 1024)
        return rif_mb + local_mb

    def reset(self) -> None:
        if self.is_initialized and self.key_rif is not None:
            self.key_rif.reset()
            self.val_rif.reset()
            self.local_keys = self.local_keys[:, :, :0, :]
            self.local_values = self.local_values[:, :, :0, :]
        self.seen_tokens = 0


class KalpanaHybridCache(Cache):
    """
    Drop-in Hybrid Cache: Exact local sliding window + O(1) RIF long-range memory.
    """
    def __init__(
        self,
        num_layers: Optional[int] = None,
        sliding_window: int = 128,
        bands: int = 4096,
        kappa: float = 1.0,
        **kwargs,
    ):
        self.sliding_window = sliding_window
        self.bands = bands
        self.kappa = kappa

        if _HAS_TRANSFORMERS_CACHE:
            if num_layers is not None:
                layers = [KalpanaHybridCacheLayer(sliding_window=sliding_window, bands=bands, kappa=kappa) for _ in range(num_layers)]
                super().__init__(layers=layers)
            else:
                class HybridLayerFactory(KalpanaHybridCacheLayer):
                    def __init__(self, *args, **kw):
                        super().__init__(sliding_window=sliding_window, bands=bands, kappa=kappa)
                super().__init__(layer_class_to_replicate=HybridLayerFactory)
        else:
            self.layers = [KalpanaHybridCacheLayer(sliding_window=sliding_window, bands=bands, kappa=kappa) for _ in range(num_layers or 32)]

    def get_total_memory_mb(self) -> float:
        total = 0.0
        for layer in self.layers:
            if hasattr(layer, "memory_footprint_mb"):
                total += layer.memory_footprint_mb()
        return total


# Alias for backwards compatibility
KalpanaKVCache = KalpanaDynamicCache
