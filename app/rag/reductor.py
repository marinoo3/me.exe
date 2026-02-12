import os

from app.config import settings

import numpy as np
from openTSNE import TSNE, TSNEEmbedding
import joblib


class Reductor:

    model_name = "tsne.joblib"

    def __init__(self) -> None:
        self._model_path = self.__make_model_path()
        self._model = self.__load_model()

    def __make_model_path(self) -> str:
        path = settings.DATA_PATH
        return os.path.join(path, 'models')

    def __load_model(self) -> TSNEEmbedding|None:
        try:
            path = os.path.join(self._model_path, self.model_name)
            return joblib.load(path)
        except FileNotFoundError:
            return None
        
    def __save_model(self, model: TSNEEmbedding) -> None:
        path = os.path.join(self._model_path, self.model_name)
        joblib.dump(model, path)
        self._model = model

    def fit_transform(self, X: np.ndarray, n = 3, p = 30) -> np.ndarray:
        """
        Fit a TSNE model, reduce the 384d vectors to 3d space and save the model

        Args:
            X (np.ndarray): List of vectors to reduce
            n (int): Number fo components. Default to 3
            p (int): Model perplexity. Default to 30

        Returns:
            np.ndarray: 3d reductions
        """
        model = TSNE(
            n_components=n, 
            metric='cosine', 
            perplexity=max(p, 1),
            n_jobs=-1
        )

        X_reduced = model.fit(X)
        self.__save_model(X_reduced)

        return np.asarray(X_reduced, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce a 384d vector to 3d space using fitted model

        Args:
            X (np.ndarray): Vector to reduce

        Returns:
            np.ndarray: 3d reduction
        """
        if self._model is None:
            raise Exception("Model must be fitted before transforming embeddings")
        
        # Transform X if only 1 dimension
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_reduced = self._model.transform(X)
        return np.asarray(X_reduced, dtype=np.float32)