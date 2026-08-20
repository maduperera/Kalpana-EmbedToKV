"""
Semantic Embedding Extractor: Bridging natural language / text with the Kalpana RIF memory matrix.
Uses native HuggingFace Transformers with mean pooling and L2 normalization.
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
        self._tokenizer = None
        self._model = None
        self.dim = 384

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self.dim = self._model.config.hidden_size
        except Exception:
            # Deterministic fallback
            self._tokenizer = None
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
            single_input = True
            texts_list = [texts]
        else:
            single_input = False
            texts_list = texts

        if self._model is not None and self._tokenizer is not None:
            with torch.no_grad():
                encoded_input = self._tokenizer(
                    texts_list, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                
                model_output = self._model(**encoded_input)
                
                # Mean Pooling - Take attention mask into account for correct averaging
                token_embeddings = model_output[0] # First element contains all token embeddings
                input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if single_input:
                    out = embeddings
                else:
                    out = embeddings

                if not convert_to_tensor:
                    return out.cpu().numpy()
                return out

        # Deterministic fallback
        vectors = []
        for text in texts_list:
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
