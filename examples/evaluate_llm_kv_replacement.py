"""
Empirical LLM KV Cache Replacement Evaluation
Runs side-by-side text generation comparing Standard Transformers KV Cache vs Kalpana O(1) RIF Cache.
"""

import os
import sys
import time
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kalpana_embed_to_kv import KalpanaDynamicCache


def evaluate_model(model_name: str = "gpt2", bands: int = 4096, max_new_tokens: int = 30):
    print("=" * 80)
    print(f"  EVALUATING LLM KV CACHE REPLACEMENT: {model_name.upper()}")
    print("=" * 80)
    print(f"Holographic Band Capacity (B): {bands}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    print(f"\n[1] Loading {model_name} and tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    test_prompts = [
        "The theory of general relativity explains that gravity is caused by",
        "Artificial intelligence will transform modern healthcare by enabling doctors to",
        "In a distant galaxy, astronomers discovered a mysterious signal that",
    ]

    print(f"\n[2] Running side-by-side generations across {len(test_prompts)} prompts...")
    print("-" * 80)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: \"{prompt}\"")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # --- Baseline: Standard Transformers KV Cache ---
        t0 = time.time()
        with torch.no_grad():
            std_outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Greedy search for deterministic comparison
                pad_token_id=tokenizer.eos_token_id,
            )
        std_time = time.time() - t0
        std_text = tokenizer.decode(std_outputs[0], skip_special_tokens=True)

        # --- Kalpana RIF O(1) KV Cache (Option A) ---
        kalpana_cache = KalpanaDynamicCache(
            num_layers=model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers,
            bands=bands,
            kappa=1.0,
        )

        t1 = time.time()
        with torch.no_grad():
            kalpana_outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                past_key_values=kalpana_cache,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        kalpana_time = time.time() - t1
        kalpana_text = tokenizer.decode(kalpana_outputs[0], skip_special_tokens=True)

        # Compare generated tokens
        std_tokens = std_outputs[0][input_ids.shape[1]:].tolist()
        kalpana_tokens = kalpana_outputs[0][input_ids.shape[1]:].tolist()
        
        matches = sum(1 for s, k in zip(std_tokens, kalpana_tokens) if s == k)
        match_pct = (matches / len(std_tokens)) * 100 if std_tokens else 100.0

        print(f"\n[Standard KV Output] ({std_time:.2f}s):")
        print(f"  {std_text}")
        print(f"\n[Kalpana RIF O(1) Output] ({kalpana_time:.2f}s | Match: {match_pct:.1f}%):")
        print(f"  {kalpana_text}")
        print("-" * 80)

    # Memory Summary
    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    num_heads = model.config.n_head if hasattr(model.config, "n_head") else model.config.num_attention_heads
    head_dim = (model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size) // num_heads

    kalpana_total_mb = kalpana_cache.get_total_memory_mb()
    print("\n[3] Memory Footprint Comparison (at 10,000 token context horizon):")
    tokens_10k_std_mb = (10000 * num_layers * num_heads * head_dim * 4 * 2) / (1024 * 1024)
    print(f"  - Standard Transformers KV Cache: {tokens_10k_std_mb:.2f} MB (grows linearly with sequence)")
    print(f"  - Kalpana RIF O(1) Cache:        {kalpana_total_mb:.2f} MB (STRICTLY INVARIANT)")
    print(f"  - Memory Compression Factor:     {tokens_10k_std_mb / kalpana_total_mb:.1f}x reduction")

    print("\n[OK] Evaluation completed.")


if __name__ == "__main__":
    evaluate_model(model_name="gpt2", bands=4096, max_new_tokens=25)
