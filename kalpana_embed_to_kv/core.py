"""
Kalpanā Enterprise Memory Layer.
Powered exclusively by the compiled Kalpanā Native Kernel.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from kalpana.core import KalpanaEngineTensor as KalpanaRIFTensor
except ImportError:
    try:
        from kalpana.core import KalpanaRIFTensor
    except ImportError:
        class KalpanaRIFTensor:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Kalpanā Native Kernel is required to initialize KalpanaRIFTensor. "
                    "Please install kalpana_sdk_enterprise."
                )


class EmbedToKVMatrix(nn.Module):
    """Semantic embedding to KV transformation adapter."""
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = embeddings.shape
        k = self.k_proj(embeddings).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(embeddings).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v
