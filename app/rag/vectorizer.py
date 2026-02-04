"""
Vectorizer Module.

This module provides text embedding generation and chunking functionality
using sentence-transformers models.

Example:
    >>> from app.services.embedding_service import Vectorizer
    >>> service = Vectorizer()
    >>> chunks = service.chunk_text("Long document text...")
    >>> embeddings = service.generate_embeddings(chunks)
"""

import os
import numpy as np
from typing import Optional

from sentence_transformers import SentenceTransformer


class Vectorizer:
    """
    Class for text embedding generation and chunking.

    This class handles loading the embedding model and provides methods
    for text chunking and embedding generation.

    Attributes:
        model_name (str): Name of the sentence-transformer model.
        chunk_size (int): Size of text chunks in characters.
        chunk_overlap (int): Overlap between consecutive chunks.
    """

    _model: Optional[SentenceTransformer] = None
    _current_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: int = 750,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Vectorizer.

        Args:
            model_name (str, optional): Sentence-transformer model name.
                Defaults to EMBEDDING_MODEL env var or "all-MiniLM-L6-v2".
            chunk_size (int): Number of characters per chunk. Defaults to 750.
            chunk_overlap (int): Overlap between chunks. Defaults to 50.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        try:
            Vectorizer._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            print(f"Error loading embedding model '{self.model_name}': {e}")
            raise

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy initialization, singleton).

        The model is shared across all Vectorizer instances to
        avoid loading it multiple times.

        Returns:
            SentenceTransformer: The loaded model instance.
        """
        if (
            Vectorizer._model is None
            or Vectorizer._current_model_name != self.model_name
        ):
            try:
                Vectorizer._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as e:
                print(f"Error loading embedding model '{self.model_name}': {e}")
                raise

            Vectorizer._current_model_name = self.model_name

        return Vectorizer._model

    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text string.

        Args:
            text (str): Text strings to embed.

        Returns:
            np.ndarray: Embedding vector.
        """
        if not text:
            raise ValueError("Can't embed empty string")

        embeddings = self.model.encode(text, show_progress_bar=False)
        return np.array(embeddings.tolist(), dtype=np.float32)

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk text into smaller pieces with overlapping segments to maintain context.

        Args:
            text (str): The text to chunk.

        Returns:
            list[str]: List of text chunks.
        """
        if not text:
            raise ValueError("Can't chunk empty string")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks
