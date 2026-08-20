"""
Semantic Embedding Extractor: Bridging natural language / text with the Kalpana RIF memory matrix.
"""

from typing import List, Optional, Union
import numpy as np
import torch


class EmbeddingExtractor:
    """
    Semantic Extractor translating human language and documents into mathematical vector representations.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self.dim: Optional[int] = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            # Fallback mock/hash embedder if sentence-transformers is not yet installed in the environment
            self._model = None
            self.dim = 384

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a single text string or list of text strings into normalized high-dimensional vectors.
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                device=self.device,
            )
            return embeddings

        # Fallback deterministic pseudo-semantic vector generator
        vectors = []
        for text in texts:
            # Deterministic seed from text hash
            seed = sum(ord(c) for c in text) % (2**31 - 1)
            g = torch.Generator().manual_seed(seed)
            vec = torch.randn(self.dim, generator=g)
            if normalize_embeddings:
                vec = torch.nn.functional.normalize(vec, p=2, dim=0)
            vectors.append(vec)

        tensor_out = torch.stack(vectors).to(self.device)
        if not convert_to_tensor:
            return tensor_out.cpu().numpy()
        return tensor_out
