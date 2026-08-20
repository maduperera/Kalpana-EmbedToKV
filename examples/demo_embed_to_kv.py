import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from kalpana_embed_to_kv import KalpanaRIFTensor, EmbeddingExtractor


def main():
    print("=" * 70)
    print("  KALPANA EmbedToKV: O(1) Holographic Semantic Memory Demo")
    print("=" * 70)

    # 1. Initialize the Semantic Vector Extractor (384-dimensional)
    print("\n[1] Initializing Semantic Extractor (all-MiniLM-L6-v2)...")
    extractor = EmbeddingExtractor()
    embed_dim = 384
    bands = 2048

    # 2. Initialize Kalpana RIF Engine for KV Storage
    print(f"[2] Initializing Kalpana RIF Tensor (Bands={bands}, Dim={embed_dim})...")
    kalpana_kv = KalpanaRIFTensor(
        batch_size=1,
        num_heads=1,
        bands=bands,
        dim=embed_dim,
        device="cpu"
    )
    print(f"    Fixed Memory Footprint: {kalpana_kv.memory_footprint_mb():.2f} MB (Permanent O(1))")

    # 3. Ingest documents into temporal holographic coordinates
    documents = [
        "Albert Einstein proposed the theory of general relativity in 1915.",
        "The Apollo 11 mission landed humans on the Moon on July 20, 1969.",
        "Photosynthesis in plants converts sunlight, water, and CO2 into glucose and oxygen.",
        "Quantum entanglement allows particles to share states across vast distances instantaneously.",
        "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.",
    ]

    print("\n[3] Ingesting knowledge snippets into RIF matrix...")
    for t, doc in enumerate(documents):
        vec = extractor.encode(doc, convert_to_tensor=True) # shape: [1, 384]
        kalpana_kv.write(t, vec)
        print(f"    -> [t={t}] Ingested: \"{doc[:55]}...\"")

    print(f"\n    Total Documents Stored: {len(documents)}")
    print(f"    Memory Footprint after storage: {kalpana_kv.memory_footprint_mb():.2f} MB (Unchanged!)")

    # 4. Search and Query against the Holographic Substrate
    test_queries = [
        "Who explained general relativity and gravity?",
        "When did astronauts first walk on the moon?",
        "How do green plants generate oxygen from sunlight?",
        "What space observatory is located at Lagrange point 2?",
    ]

    print("\n[4] Querying Holographic Memory with Natural Language:")
    print("-" * 70)

    for query in test_queries:
        query_vec = extractor.encode(query, convert_to_tensor=True) # [1, 384]
        
        # Sweep all temporal coordinates and calculate resonance (cosine similarity of reconstructed vectors)
        t_range = torch.arange(0, len(documents)).float()
        # past_vectors: [1, 1, num_docs, 384]
        past_vectors = kalpana_kv.batch_reconstruct(t_range).squeeze(0).squeeze(0) # [num_docs, 384]
        
        # Normalize and compute dot-product resonance
        past_norm = torch.nn.functional.normalize(past_vectors, p=2, dim=-1)
        q_norm = torch.nn.functional.normalize(query_vec.squeeze(0), p=2, dim=-1)
        scores = torch.matmul(past_norm, q_norm) # [num_docs]
        
        best_t = int(torch.argmax(scores).item())
        best_score = float(scores[best_t].item())

        print(f"  Query    : \"{query}\"")
        print(f"  Top Match: [t={best_t}] (Resonance: {best_score:.4f})")
        print(f"  Content  : \"{documents[best_t]}\"")
        print("-" * 70)

    print("\n[OK] Demo completed successfully.")


if __name__ == "__main__":
    main()
