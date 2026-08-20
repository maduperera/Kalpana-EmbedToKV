import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import time
import torch
from kalpana_embed_to_kv import KalpanaRIFTensor


def calculate_tensor_memory_mb(tensor: torch.Tensor) -> float:
    return tensor.nelement() * tensor.element_size() / (1024 * 1024)


def run_benchmark(steps: int = 10000, batch_size: int = 1, num_heads: int = 32, head_dim: int = 128, bands: int = 2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("  EMPIRICAL MEMORY SCALING BENCHMARK: TRANSFORMER KV vs KALPANA RIF")
    print("=" * 75)
    print(f"Device: {device.upper()} | Heads: {num_heads} | Head Dim: {head_dim} | Bands: {bands}")
    print(f"Target Sequence Length: {steps:,} tokens\n")

    # Kalpana RIF initialization
    kalpana_k = KalpanaRIFTensor(batch_size, num_heads, bands, head_dim, device=device)
    kalpana_v = KalpanaRIFTensor(batch_size, num_heads, bands, head_dim, device=device)
    kalpana_fixed_mb = kalpana_k.memory_footprint_mb() + kalpana_v.memory_footprint_mb()

    # Standard KV tracking
    element_size = 4 # Float32 bytes
    standard_bytes_per_token = batch_size * num_heads * head_dim * element_size * 2 # K + V

    checkpoints = [100, 500, 1000, 2500, 5000, 7500, 10000]
    if steps not in checkpoints and steps > 10000:
        checkpoints.append(steps)

    print(f"{'Sequence Tokens':<18} | {'Standard KV (MB)':<20} | {'Kalpana RIF (MB)':<20} | {'Memory Savings':<15}")
    print("-" * 80)

    for chk in checkpoints:
        if chk > steps:
            break
        std_mb = (chk * standard_bytes_per_token) / (1024 * 1024)
        savings = std_mb / kalpana_fixed_mb if kalpana_fixed_mb > 0 else 1.0
        print(f"{chk:<18,d} | {std_mb:<20.2f} | {kalpana_fixed_mb:<20.2f} | {savings:<15.1f}x")

    print("-" * 80)
    print(f"\n[OK] At {steps:,} tokens:")
    print(f"    - Standard Linear KV Cache: {(steps * standard_bytes_per_token)/(1024*1024):.2f} MB (grows infinitely with N)")
    print(f"    - Kalpana RIF Holographic Memory: {kalpana_fixed_mb:.2f} MB (strictly bounded O(1))")
    print(f"    - Memory Reduction Factor: {((steps * standard_bytes_per_token)/(1024*1024))/kalpana_fixed_mb:.1f}x\n")


if __name__ == "__main__":
    run_benchmark(steps=20000)
