"""
Needle-in-a-Haystack Long-Context Retrieval Benchmark for Kalpana EmbedToKV
Tests accurate retrieval of specific hidden facts buried deep in multi-thousand token contexts.
"""

import os
import sys
import time
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from kalpana_embed_to_kv import KalpanaRIFTensor, EmbeddingExtractor


def generate_haystack_corpus(num_chunks: int = 500):
    background_templates = [
        "Atmospheric turbulence in planetary atmospheres causes convective heat transfer.",
        "Silicon semiconductor manufacturing requires cleanroom environments with sub-micron filtration.",
        "Deep convolutional networks utilize residual skip connections to mitigate gradient vanishing.",
        "Hydrodynamic cavitation creates localized shockwaves damaging marine propeller surfaces.",
        "Superconducting quantum interference devices detect minute magnetic flux fluctuations.",
        "Enzymatic catalysts accelerate biological reactions by lowering the activation energy barrier.",
        "Geothermal energy systems extract thermodynamic enthalpy from subsurface volcanic aquifers.",
    ]
    haystack = []
    for i in range(num_chunks):
        base = background_templates[i % len(background_templates)]
        haystack.append(f"[Chunk {i}] {base} Sequence variation {i * 7 + 13}.")
    return haystack


def run_needle_benchmark(context_chunks: int = 1000, bands: int = 4096, dim: int = 384):
    print("=" * 80)
    print("  KALPANA RIF: NEEDLE-IN-A-HAYSTACK RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"Context Horizon: {context_chunks} chunks (~{context_chunks * 25:,} tokens)")
    print(f"Holographic Band Capacity: {bands} | Vector Dim: {dim}")

    extractor = EmbeddingExtractor()
    rif = KalpanaRIFTensor(batch_size=1, num_heads=1, bands=bands, dim=dim, kappa=1.0)

    # Prepare Haystack
    haystack = generate_haystack_corpus(context_chunks)

    # Insert 3 Target Needles at 10%, 50%, and 90% depth
    needles = [
        (int(context_chunks * 0.10), "The secret passkey for Project Chronos is OMEGA-7749.", "What is the secret passkey for Project Chronos?"),
        (int(context_chunks * 0.50), "Dr. Elena Vance invented the resonant hyper-drive in Neo-Geneva.", "Who invented the resonant hyper-drive?"),
        (int(context_chunks * 0.90), "The emergency shutdown code for reactor 4 is EPSILON-9021.", "What is the emergency shutdown code for reactor 4?"),
    ]

    for idx, needle_text, _ in needles:
        haystack[idx] = needle_text

    print(f"\n[1] Ingesting {context_chunks:,} chunks into O(1) Holographic Memory...")
    t0 = time.time()
    for t, chunk in enumerate(haystack):
        vec = extractor.encode(chunk, convert_to_tensor=True)
        rif.write(t, vec)
    t_ingest = time.time() - t0

    print(f"    Ingestion completed in {t_ingest:.2f}s ({context_chunks / t_ingest:.1f} chunks/sec)")
    print(f"    Total Memory Used: {rif.memory_footprint_mb():.2f} MB (STRICTLY CONSTANT O(1))")

    print("\n[2] Executing Needle Retrieval Queries via Holographic Sweep:")
    print("-" * 80)

    correct = 0
    t_range = torch.arange(0, len(haystack)).float()
    past_vectors = rif.batch_reconstruct(t_range).squeeze() # [num_chunks, dim]
    past_norm = torch.nn.functional.normalize(past_vectors, dim=-1)

    for depth_idx, needle_text, query in needles:
        q_vec = extractor.encode(query, convert_to_tensor=True)
        q_norm = torch.nn.functional.normalize(q_vec.squeeze(0), dim=-1)

        scores = torch.matmul(past_norm, q_norm)
        top_t = int(torch.argmax(scores).item())
        top_score = float(scores[top_t].item())

        is_hit = (top_t == depth_idx)
        if is_hit:
            correct += 1

        print(f"Query: \"{query}\"")
        print(f"  Target Coordinate : [t={depth_idx}] (Depth: {(depth_idx/context_chunks)*100:.1f}%)")
        print(f"  Retrieved Coord   : [t={top_t}] (Resonance Score: {top_score:.4f})")
        print(f"  Result Status     : {'[PASSED - EXACT HIT]' if is_hit else '[FAILED]'}")
        print(f"  Retrieved Chunk   : \"{haystack[top_t]}\"")
        print("-" * 80)

    accuracy_pct = (correct / len(needles)) * 100
    print(f"\n[3] Benchmark Summary:")
    print(f"    - Needles Retrieved: {correct}/{len(needles)} ({accuracy_pct:.1f}% Accuracy)")
    print(f"    - Context Scale: {len(haystack):,} chunks")
    print(f"    - Memory Footprint: {rif.memory_footprint_mb():.2f} MB")
    print("\n[OK] Needle-in-a-Haystack benchmark completed.")


if __name__ == "__main__":
    run_needle_benchmark(context_chunks=500, bands=4096)
