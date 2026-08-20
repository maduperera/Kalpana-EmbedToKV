import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from kalpana_embed_to_kv import KalpanaKVCache, KalpanaDynamicCache


def simulate_huggingface_layer_generation():
    print("=" * 70)
    print("  KALPANA DynamicCache: HuggingFace LLM Simulation")
    print("=" * 70)

    NUM_LAYERS = 4
    NUM_HEADS = 8
    HEAD_DIM = 64
    BANDS = 1024
    SEQ_LEN_PROMPT = 32
    GEN_TOKENS = 64

    print(f"Config: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, HeadDim={HEAD_DIM}, Bands={BANDS}")
    
    # 1. Initialize Kalpana Dynamic Cache
    cache = KalpanaDynamicCache(
        num_layers=NUM_LAYERS,
        bands=BANDS,
    )
    # Lazy initial state warmup
    k_warmup = torch.zeros(1, NUM_HEADS, 1, HEAD_DIM)
    v_warmup = torch.zeros(1, NUM_HEADS, 1, HEAD_DIM)
    for layer_idx in range(NUM_LAYERS):
        cache.layers[layer_idx].lazy_initialization(k_warmup, v_warmup)

    print(f"Kalpana Total Cache Size: {cache.get_total_memory_mb():.2f} MB (Fixed across all {NUM_LAYERS} layers)")

    # 2. Ingest Prompt Key/Value states
    print(f"\n[Step 1] Ingesting Prompt ({SEQ_LEN_PROMPT} tokens)...")
    for layer_idx in range(NUM_LAYERS):
        k_prompt = torch.randn(1, NUM_HEADS, SEQ_LEN_PROMPT, HEAD_DIM)
        v_prompt = torch.randn(1, NUM_HEADS, SEQ_LEN_PROMPT, HEAD_DIM)
        past_k, past_v = cache.update(k_prompt, v_prompt, layer_idx=layer_idx)

    print(f"Cache sequence length after prompt: {cache.get_seq_length()} tokens")

    # 3. Simulate Auto-Regressive Token Generation Loop
    print(f"\n[Step 2] Generating {GEN_TOKENS} new tokens auto-regressively...")
    for step in range(GEN_TOKENS):
        for layer_idx in range(NUM_LAYERS):
            k_new = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
            v_new = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
            past_k, past_v = cache.update(k_new, v_new, layer_idx=layer_idx)

    print(f"Total Sequence Length processed: {cache.get_seq_length()} tokens")
    print(f"Final Memory Size: {cache.get_total_memory_mb():.2f} MB")
    print("\n[OK] HuggingFace DynamicCache simulation verified successfully!")


if __name__ == "__main__":
    simulate_huggingface_layer_generation()
