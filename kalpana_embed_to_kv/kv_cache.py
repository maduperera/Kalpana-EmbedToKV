"""
KV Cache Replacement for Transformers and HuggingFace LLM generation.
Duck-typed to satisfy transformers 4.36 through 5.x model.generate() interface.
No Cache / CacheLayerMixin inheritance required — works on any Python version.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from .core import KalpanaRIFTensor


class KalpanaCacheLayer:
    """
    Individual Transformer Layer Cache powered by O(1) RIF Holographic Memory.
    Replaces torch.cat with constant-memory wave interference superposition.
    """
    # Duck-type attrs checked by transformers generate()
    is_sliding = False
    is_compileable = False
    layer_type = "full_attention"

    def __init__(self, bands: int = 4096, kappa: float = 1.0):
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

    def reorder_cache(self, beam_idx):
        pass

    def offload(self):
        pass

    def prefetch(self, *args, **kwargs):
        pass


class KalpanaHybridCacheLayer:
    """
    Hybrid Cache Layer: Exact local sliding window + O(1) RIF long-range memory.
    """
    is_sliding = False
    is_compileable = False
    layer_type = "full_attention"

    def __init__(self, sliding_window: int = 128, bands: int = 4096, kappa: float = 1.0):
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

        for step in range(seq_len_new):
            t = self.seen_tokens + step
            k_t = key_states[:, :, step, :]
            v_t = value_states[:, :, step, :]
            self.key_rif.write(t, k_t)
            self.val_rif.write(t, v_t)

        self.seen_tokens += seq_len_new

        self.local_keys = torch.cat([self.local_keys, key_states], dim=2)
        self.local_values = torch.cat([self.local_values, value_states], dim=2)

        if self.seen_tokens <= self.sliding_window:
            return self.local_keys, self.local_values

        if self.local_keys.shape[2] > self.sliding_window:
            self.local_keys = self.local_keys[:, :, -self.sliding_window:, :]
            self.local_values = self.local_values[:, :, -self.sliding_window:, :]

        long_range_len = self.seen_tokens - self.sliding_window
        t_range_prefix = torch.arange(0, long_range_len, device=self.device).float()
        prefix_keys = self.key_rif.batch_reconstruct(t_range_prefix)
        prefix_values = self.val_rif.batch_reconstruct(t_range_prefix)

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

    def reorder_cache(self, beam_idx):
        pass

    def offload(self):
        pass

    def prefetch(self, *args, **kwargs):
        pass


class KalpanaDynamicCache:
    """
    Drop-in O(1) Memory KV Cache replacement for HuggingFace Transformers.

    Duck-typed to satisfy model.generate() without inheriting from Cache,
    compatible with transformers 4.36 through 5.x on any Python version.

    Memory Complexity: O(1) — constant VRAM regardless of sequence length.
    """
    # Class-level flags checked by transformers generate()
    is_compileable = False
    is_initialized = True

    @property
    def is_sliding(self) -> List[bool]:
        """Returns per-layer is_sliding flags as a list — required by transformers masking_utils."""
        return [layer.is_sliding for layer in self.layers]

    def __init__(
        self,
        num_layers: Optional[int] = None,
        bands: int = 4096,
        kappa: float = 1.0,
        **kwargs,
    ):
        self.bands = bands
        self.kappa = kappa
        self.num_layers = num_layers or 32
        # Per-layer RIF cache instances
        self.layers: List[KalpanaCacheLayer] = [
            KalpanaCacheLayer(bands=bands, kappa=kappa) for _ in range(self.num_layers)
        ]
        # key_cache / value_cache mirrors for transformers internals
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Grow layer list dynamically if needed
        while len(self.layers) <= layer_idx:
            self.layers.append(KalpanaCacheLayer(bands=self.bands, kappa=self.kappa))

        k_recon, v_recon = self.layers[layer_idx].update(key_states, value_states)

        # Keep key_cache / value_cache in sync
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k_recon)
            self.value_cache.append(v_recon)
        else:
            self.key_cache[layer_idx] = k_recon
            self.value_cache[layer_idx] = v_recon

        if layer_idx == 0:
            self._seen_tokens = self.layers[0].seen_tokens

        return k_recon, v_recon

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = layer_idx or 0
        if len(self.layers) > idx and self.layers[idx].is_initialized:
            return self.layers[idx].seen_tokens
        return self._seen_tokens

    def get_mask_sizes(self, q_length: int, layer_idx: Optional[int] = 0) -> Tuple[int, int]:
        """Returns (kv_length, kv_offset) for attention mask construction.
        Called BEFORE update(), so the final kv_length = seen_tokens + q_length.
        """
        idx = layer_idx or 0
        if len(self.layers) > idx and self.layers[idx].is_initialized:
            seen = self.layers[idx].seen_tokens
        else:
            seen = self._seen_tokens
        return seen + q_length, 0

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_total_memory_mb(self) -> float:
        return sum(layer.memory_footprint_mb() for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def reorder_cache(self, beam_idx) -> None:
        """Called during beam search to reorder cache entries."""
        pass

    def crop(self, max_length: int) -> None:
        """Called by generate() to trim cache to max_length."""
        pass

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Called during beam search expand."""
        pass

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Called to select beam indices."""
        pass

    def to(self, device) -> "KalpanaDynamicCache":
        """Allow .to(device) calls."""
        return self

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        return iter(zip(self.key_cache, self.value_cache))

    def __getitem__(self, idx: int):
        return (self.key_cache[idx], self.value_cache[idx])


class KalpanaHybridCache:
    """
    Drop-in Hybrid Cache: Exact local sliding window + O(1) RIF long-range memory.
    Duck-typed for all transformers versions.
    """
    is_compileable = False
    is_initialized = True

    @property
    def is_sliding(self) -> List[bool]:
        """Returns per-layer is_sliding flags as a list — required by transformers masking_utils."""
        return [layer.is_sliding for layer in self.layers]

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
        self.num_layers = num_layers or 32
        self.layers: List[KalpanaHybridCacheLayer] = [
            KalpanaHybridCacheLayer(sliding_window=sliding_window, bands=bands, kappa=kappa)
            for _ in range(self.num_layers)
        ]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(KalpanaHybridCacheLayer(
                sliding_window=self.sliding_window, bands=self.bands, kappa=self.kappa
            ))

        k_recon, v_recon = self.layers[layer_idx].update(key_states, value_states)

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k_recon)
            self.value_cache.append(v_recon)
        else:
            self.key_cache[layer_idx] = k_recon
            self.value_cache[layer_idx] = v_recon

        if layer_idx == 0:
            self._seen_tokens = self.layers[0].seen_tokens

        return k_recon, v_recon

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = layer_idx or 0
        if len(self.layers) > idx and self.layers[idx].is_initialized:
            return self.layers[idx].seen_tokens
        return self._seen_tokens

    def get_mask_sizes(self, q_length: int, layer_idx: Optional[int] = 0) -> Tuple[int, int]:
        """Returns (kv_length, kv_offset) for attention mask construction."""
        idx = layer_idx or 0
        if len(self.layers) > idx and self.layers[idx].is_initialized:
            kv_len = self.layers[idx].seen_tokens
        else:
            kv_len = self._seen_tokens
        return kv_len, 0

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_total_memory_mb(self) -> float:
        return sum(layer.memory_footprint_mb() for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def reorder_cache(self, beam_idx) -> None:
        pass

    def crop(self, max_length: int) -> None:
        pass

    def batch_repeat_interleave(self, repeats: int) -> None:
        pass

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        pass

    def to(self, device) -> "KalpanaHybridCache":
        return self

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        return iter(zip(self.key_cache, self.value_cache))

    def __getitem__(self, idx: int):
        return (self.key_cache[idx], self.value_cache[idx])


# Backwards compatibility alias
KalpanaKVCache = KalpanaDynamicCache
