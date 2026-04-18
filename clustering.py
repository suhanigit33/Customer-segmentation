import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    inertia: Optional[float]          # KMeans only
    model: object = field(repr=False)


# ── K-Means ────────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> ClusterResult:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return _make_result(labels, X, model, inertia=model.inertia_)


def elbow_curve(X: np.ndarray, k_range=range(2, 11)) -> dict:
    """Return inertia and silhouette for each k."""
    inertias, silhouettes = [], []
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(m.inertia_)
        sil = silhouette_score(X, m.labels_) if k > 1 else None
        silhouettes.append(sil)
    return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def run_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 5) -> ClusterResult:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return _make_result(labels, X, model)


# ── Hierarchical ───────────────────────────────────────────────────────────────

def run_hierarchical(X: np.ndarray, k: int, linkage: str = "ward") -> ClusterResult:
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return _make_result(labels, X, model)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(labels, X, model, inertia=None) -> ClusterResult:
    valid = labels[labels >= 0]
    n_clusters = len(set(valid))
    X_valid = X[labels >= 0]

    if n_clusters >= 2 and len(X_valid) > n_clusters:
        sil = float(silhouette_score(X_valid, valid))
        db  = float(davies_bouldin_score(X_valid, valid))
        ch  = float(calinski_harabasz_score(X_valid, valid))
    else:
        sil = db = ch = None

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        inertia=inertia,
        model=model,
    )


def segment_profiles(df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Compute per-segment mean for each feature, plus count and share."""
    df2 = df[feature_cols].copy()
    df2["_Segment"] = labels
    df2 = df2[df2["_Segment"] >= 0]
    agg = df2.groupby("_Segment")[feature_cols].mean().round(2)
    agg["Count"] = df2.groupby("_Segment").size()
    agg["Share (%)"] = (agg["Count"] / agg["Count"].sum() * 100).round(1)
    return agg.reset_index()
