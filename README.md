# <img src="./assets/logo.svg" width="40" height="40" align="center"> Kalpana EmbedToKV

> **$O(1)$ Holographic Memory KV Cache Replacement Engine powered by Resonant Interference Field (RIF) Mathematics.**

---

## 🌟 Executive Overview

In standard Transformer and Large Language Model (LLM) architectures, the **Key-Value (KV) Cache** grows linearly ($O(N)$) with sequence length. At massive context windows (100k–1M+ tokens), the KV cache consumes tens of gigabytes of VRAM, creating the notorious **"Memory Wall"** bottleneck that prevents long-context models from running on edge devices and consumer hardware.

**Kalpana EmbedToKV** utilizes the proprietary **Kalpana SDK** WebAssembly and PyTorch mathematical engines to convert high-dimensional semantic and token embeddings directly into a **fixed-size $O(1)$ holographic memory matrix**.

![Kalpana Architecture Flow](./assets/kalpana_architecture.png)

### Key Capabilities
- **Strict $O(1)$ Memory Footprint**: Memory consumption remains 100% constant regardless of whether you process 1,000 or 10,000,000 tokens.
- **Drop-in KV Cache Replacement**: Native PyTorch `KalpanaDynamicCache` adapter compatible with HuggingFace Transformers `model.generate()`.
- **Language & Framework Agnostic**: Python PyTorch bindings for server/GPU inference and high-performance WebAssembly (`kalpana_vault.wasm`) for edge and browser environments.
- **Semantic Associative Recall**: Parallel holographic resonance sweep eliminates sequential linear attention bottlenecks.

---

## 📊 Benchmark: Standard KV vs Kalpana RIF

| Sequence Length | Standard Linear KV Cache | Kalpana RIF Engine | Memory Reduction |
| :--- | :--- | :--- | :--- |
| **1,000 tokens** | ~32 MB | **12.0 MB** | 2.7x |
| **10,000 tokens** | ~320 MB | **12.0 MB** | 26.6x |
| **100,000 tokens** | ~3.2 GB | **12.0 MB** | 266x |
| **1,000,000 tokens** | ~32.0 GB | **12.0 MB** | 2,666x |
| **3,000,000 tokens** | **~35.2 GB** | **12.0 MB** | **2,933x** |

---

## 🚀 Quickstart: Python

### 1. Installation
```bash
pip install -e .
```

### 2. Ingesting Embeddings into Holographic Memory
```python
import torch
from kalpana_embed_to_kv import KalpanaRIFTensor, EmbeddingExtractor

# 1. Initialize extractor and O(1) RIF tensor
extractor = EmbeddingExtractor()
rif_memory = KalpanaRIFTensor(batch_size=1, num_heads=1, bands=2048, dim=384)

# 2. Ingest semantic document chunks
documents = [
    "Albert Einstein formulated general relativity in 1915.",
    "Apollo 11 landed humans on the Moon in July 1969.",
    "Photosynthesis converts sunlight into glucose and oxygen."
]

for t, doc in enumerate(documents):
    vec = extractor.encode(doc)
    rif_memory.write(t, vec)

# 3. Holographic Resonance Sweep
query_vec = extractor.encode("When did astronauts walk on the moon?")
t_range = torch.arange(0, len(documents)).float()
past_vectors = rif_memory.batch_reconstruct(t_range).squeeze()

# Compute resonance
scores = torch.matmul(
    torch.nn.functional.normalize(past_vectors, dim=-1),
    torch.nn.functional.normalize(query_vec.squeeze(0), dim=-1)
)
best_match_idx = torch.argmax(scores).item()
print("Matched Document:", documents[best_match_idx])
```

### 3. HuggingFace KV Cache Drop-in Integration
```python
from kalpana_embed_to_kv import KalpanaDynamicCache

# Initialize Kalpana Dynamic Cache for multi-layer LLM
kalpana_cache = KalpanaDynamicCache(
    num_layers=32,
    batch_size=1,
    num_heads=32,
    head_dim=128,
    bands=2048
)

# Pass directly to model generation
# outputs = model.generate(input_ids, past_key_values=kalpana_cache)
```

---

## 🌐 Quickstart: WebAssembly & JavaScript

```javascript
import { KalpanaVaultEmbedToKV } from './pkg_vault/kalpana_vault_embed.js';

// Initialize WASM RIF engine
const vault = new KalpanaVaultEmbedToKV({
  bands: 2048,
  dim: 384,
  wasmPath: './pkg_vault/kalpana_vault.wasm'
});

await vault.initialize();

// Ingest embedding vector
const t = vault.ingestEmbedding(embeddingArray, { title: "Research Note" });

// Holographic search
const results = vault.search(queryEmbeddingArray, 5);
console.log("Top Resonance Match:", results[0]);
```

---

## 📁 Repository Structure

```
Kalpana-EmbedToKV/
├── kalpana_embed_to_kv/             # Core Python Package
│   ├── __init__.py                  # Package exports
│   ├── core.py                      # KalpanaRIFTensor & EmbedToKVMatrix
│   ├── kv_cache.py                  # KalpanaKVCache & KalpanaDynamicCache
│   ├── attention.py                 # KalpanaAttentionLayer & KV Interpreter
│   └── extractor.py                 # Semantic vector embedding bridge
├── pkg_vault/                       # Core WebAssembly Engine from Kalpana-SDK
│   ├── kalpana_vault.js             # WASM loader & glue code
│   ├── kalpana_vault.wasm           # Compiled RIF engine binary
│   └── kalpana_vault_embed.js       # High-level JS/TS Embed-to-KV module
├── examples/                        # Working examples & benchmarks
│   ├── demo_embed_to_kv.py          # End-to-end semantic text to KV & recall
│   ├── demo_huggingface_kv.py       # HuggingFace multi-layer dynamic cache test
│   └── benchmark_memory_scaling.py  # 3M token empirical memory scaling test
├── web_demo/                        # Interactive Browser-Native Demo
│   ├── index.html                   # Modern glassmorphic Web UI
│   └── app.js                       # Client-side embedding & sweep controller
├── assets/                          # Architecture diagrams & branding
├── tests/                           # Unit test suite
├── setup.py                         # Python package setup
├── package.json                     # NPM package configuration
└── README.md
```

---

## 🔒 Intellectual Property & Patent Notice
**Patent Pending Application No. LK/P/1/24089**  
*Proprietary and Confidential Technology by Vijñāna AI.*  
Unauthorized duplication or reverse-engineering of the mathematical logic within the `kalpana_vault.wasm` binary or RIF matrix formulations is strictly prohibited.

For commercial licensing and enterprise inquiries:  
📧 **support@vijnanaai.com** | **Vijñāna AI — Intelligence, Redefined.**
