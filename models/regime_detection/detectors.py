from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle

# On Windows + MKL, scikit-learn may warn about KMeans memory leak if thread count is too high.
# Set thread caps before numerical libraries are imported.
if os.name == "nt":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectorConfig:
    n_states: int = 3
    random_state: int = 42


class BaseDetector:
    def __init__(self, name: str, n_states: int = 3, random_state: int = 42) -> None:
        self.name = name
        self.n_states = n_states
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, x_train: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str | Path) -> "BaseDetector":
        path = Path(model_path)
        with path.open("rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, BaseDetector):
            raise TypeError(f"File does not contain a valid detector: {path}")
        if cls is not BaseDetector and not isinstance(loaded, cls):
            raise TypeError(f"Loaded detector type mismatch for {path}: expected {cls.__name__}")
        return loaded


class HMMDetector(BaseDetector):
    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
        n_iter: int = 300,
        covariance_type: str = "diag",
    ) -> None:
        super().__init__(name="hmm", n_states=n_states, random_state=random_state)
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_input)
        states = self.model.predict(x_scaled)
        return states.astype(int)


class GMMDetector(BaseDetector):
    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
        covariance_type: str = "full",
    ) -> None:
        super().__init__(name="gmm", n_states=n_states, random_state=random_state)
        self.model = GaussianMixture(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=5,
        )

    def fit(self, x_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_input)
        states = self.model.predict(x_scaled)
        return states.astype(int)


class KMeansDetector(BaseDetector):
    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        super().__init__(name="kmeans", n_states=n_states, random_state=random_state)
        self.model = KMeans(
            n_clusters=n_states,
            random_state=random_state,
            n_init=20,
        )

    def fit(self, x_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_input)
        states = self.model.predict(x_scaled)
        return states.astype(int)
