import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y)}")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of observations, "
                f"got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        
        if X.shape[0] == 0:
            raise ValueError(f"X cannot be empty")
        
        n_samples = X.shape[0]
        labels = np.unique(y)
        
        if len(labels) < 2:
            # If there's only one cluster, silhouette score is 0 or undefined
            return np.zeros(n_samples)
        
        # Compute pairwise distances
        distances = cdist(X, X, metric='euclidean')
        
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            cluster_i = y[i]
            
            # a(i): mean distance to points in the same cluster
            same_cluster_mask = y == cluster_i
            if np.sum(same_cluster_mask) > 1:
                # Exclude the point itself
                a_i = np.mean(distances[i, same_cluster_mask & (np.arange(n_samples) != i)])
            else:
                a_i = 0.0
            
            # b(i): minimum mean distance to points in other clusters
            b_i = float('inf')
            for label in labels:
                if label != cluster_i:
                    other_cluster_mask = y == label
                    if np.sum(other_cluster_mask) > 0:
                        mean_dist = np.mean(distances[i, other_cluster_mask])
                        b_i = min(b_i, mean_dist)
            
            # Compute silhouette coefficient for this point
            if b_i == float('inf'):
                silhouette_scores[i] = 0.0
            else:
                # Handle case where both a_i and b_i are 0 (identical points)
                denominator = max(a_i, b_i)
                if denominator == 0:
                    silhouette_scores[i] = 0.0
                else:
                    s_i = (b_i - a_i) / denominator
                    silhouette_scores[i] = s_i
        
        return silhouette_scores
