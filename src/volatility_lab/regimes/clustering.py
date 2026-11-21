"""K-Means feature-based volatility clustering."""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansRegimeClusterer:
    """Multi-feature volatility state clustering (level, trend, range, and vol-of-vol)."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()

    def fit_predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """Clusters multi-dimensional volatility features into market states."""
        clean = feature_df.dropna()
        X_scaled = self.scaler.fit_transform(clean.values)
        clusters = self.kmeans.fit_predict(X_scaled)
        return pd.Series(clusters, index=clean.index, name="kmeans_regime")
