"""
Reranker Module.

This module provides text embedding generation and chunking functionality
using sentence-transformers CrossEncoder models.
"""

import os
from typing import Optional

from sentence_transformers import CrossEncoder

from app.models import Chunk


class Reranker:
    """
    Class for reranking relevant chunks using cross-encoder.

    This class handles loading the cross-encoder model and provides methods
    for rekank the chunks.

    Attributes:
        model_name (str): Name of the cross-encoder model.
    """

    _model: Optional[CrossEncoder] = None
    _current_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the Reranker.

        Args:
            model_name (str, optional): cross-encoder model name.
                Defaults to RERANKER_MODEL env var or "all-MiniLM-L6-v2".
        """
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        try:
            Reranker._model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"Error loading reranker model '{self.model_name}': {e}")
            raise

    @property
    def model(self) -> CrossEncoder:
        """
        Get or load the cross-encoder model (lazy initialization, singleton).

        The model is shared across all CrossEncoder instances to
        avoid loading it multiple times.

        Returns:
            CrossEncoder: The loaded model instance.
        """
        if (
            Reranker._model is None
            or Reranker._current_model_name != self.model_name
        ):
            try:
                Reranker._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"Error loading reranker model '{self.model_name}': {e}")
                raise

            Reranker._current_model_name = self.model_name

        return Reranker._model

    def rerank(self, query: str, chunks: list[Chunk], threeshold = -2) -> list[Chunk]:
        """
        Rerank relevant chunks based on query

        Args:
            query_emb (np.ndarray): Query
            chunks (list[Chunk]): Relevant chunks to rerank
            threeshold (int): Minimum score to keep a chunk

        Returns:
            list[Chunk]: Reranked chunks
        """
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked_chunks: list[Chunk] = []
        for score, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True):
            if score >= threeshold:
                chunk.score = float(score)
                reranked_chunks.append(chunk)
        
        return reranked_chunks
