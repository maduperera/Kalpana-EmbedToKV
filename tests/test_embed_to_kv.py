"""
Unit tests for Kalpana EmbedToKV
"""

import unittest
import torch
from kalpana_embed_to_kv.core import KalpanaRIFTensor, EmbedToKVMatrix
from kalpana_embed_to_kv.kv_cache import KalpanaKVCache, KalpanaDynamicCache
from kalpana_embed_to_kv.attention import KalpanaAttentionLayer, KalpanaKVInterpreter


class TestKalpanaEmbedToKV(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.num_heads = 4
        self.bands = 512
        self.head_dim = 64
        self.embed_dim = self.num_heads * self.head_dim

    def test_rif_tensor_memory_invariant(self):
        """Verify memory stays O(1) regardless of number of writes."""
        rif = KalpanaRIFTensor(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            bands=self.bands,
            dim=self.head_dim,
        )
        initial_mem = rif.memory_footprint_mb()
        self.assertGreater(initial_mem, 0)

        # Write 50 tokens
        for t in range(50):
            vec = torch.randn(self.batch_size, self.num_heads, self.head_dim)
            rif.write(t, vec)

        after_50_mem = rif.memory_footprint_mb()
        self.assertEqual(initial_mem, after_50_mem, "Memory footprint must remain constant O(1)")

    def test_batch_reconstruct_shapes(self):
        """Verify batch reconstruction outputs correct shapes."""
        rif = KalpanaRIFTensor(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            bands=self.bands,
            dim=self.head_dim,
        )
        seq_len = 10
        for t in range(seq_len):
            vec = torch.randn(self.batch_size, self.num_heads, self.head_dim)
            rif.write(t, vec)

        t_range = torch.arange(0, seq_len).float()
        past = rif.batch_reconstruct(t_range)
        self.assertEqual(past.shape, (self.batch_size, self.num_heads, seq_len, self.head_dim))

    def test_embed_to_kv_matrix(self):
        """Verify embedding projection and KV storage."""
        matrix = EmbedToKVMatrix(
            embed_dim=self.embed_dim,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            bands=self.bands,
        )
        emb = torch.randn(self.batch_size, self.embed_dim)
        t_idx = matrix.ingest_embedding(emb)
        self.assertEqual(t_idx, 0)

        keys, values = matrix.get_past_kv()
        self.assertEqual(keys.shape, (self.batch_size, self.num_heads, 1, self.head_dim))
        self.assertEqual(values.shape, (self.batch_size, self.num_heads, 1, self.head_dim))

    def test_attention_layer_forward(self):
        """Verify multi-head attention forward pass."""
        layer = KalpanaAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            bands=self.bands,
        )
        hidden = torch.randn(self.batch_size, 5, self.embed_dim)
        out, weights = layer(hidden)
        self.assertEqual(out.shape, (self.batch_size, 5, self.embed_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, 5, 5))

    def test_kv_cache_multi_layer(self):
        """Verify multi-layer cache updates and sequence lengths."""
        cache = KalpanaDynamicCache(
            num_layers=2,
            bands=self.bands,
        )
        k = torch.randn(self.batch_size, self.num_heads, 3, self.head_dim)
        v = torch.randn(self.batch_size, self.num_heads, 3, self.head_dim)

        past_k0, past_v0 = cache.update(k, v, layer_idx=0)
        past_k1, past_v1 = cache.update(k, v, layer_idx=1)

        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(past_k0.shape, (self.batch_size, self.num_heads, 3, self.head_dim))

    def test_hybrid_cache(self):
        """Verify hybrid cache sliding window + long range RIF prefix."""
        from kalpana_embed_to_kv import KalpanaHybridCache
        cache = KalpanaHybridCache(num_layers=1, sliding_window=4, bands=self.bands)

        # Ingest 10 tokens
        for step in range(10):
            k = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            v = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            past_k, past_v = cache.update(k, v, layer_idx=0)

        self.assertEqual(cache.get_seq_length(), 10)
        self.assertEqual(past_k.shape, (self.batch_size, self.num_heads, 10, self.head_dim))
        self.assertEqual(past_v.shape, (self.batch_size, self.num_heads, 10, self.head_dim))


if __name__ == "__main__":
    unittest.main()
