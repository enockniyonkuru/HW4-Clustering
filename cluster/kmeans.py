import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = None
        self._fitted = False

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"mat must be a numpy array, got {type(mat)}")
        if mat.ndim != 2:
            raise ValueError(f"mat must be 2D, got shape {mat.shape}")
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            raise ValueError(f"mat cannot be empty, got shape {mat.shape}")
        
        n_samples, n_features = mat.shape
        
        if self.k > n_samples:
            raise ValueError(f"k ({self.k}) cannot be greater than number of observations ({n_samples})")
        
        # Initialize centroids using k-means++ initialization
        self.centroids = self._initialize_centroids(mat)
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = cdist(mat, self.centroids, metric='euclidean')
            labels = np.argmin(distances, axis=1)
            
            # Compute error (sum of squared distances from points to assigned centroids)
            current_error = np.sum((distances[np.arange(n_samples), labels]) ** 2) / n_samples
            
            # Check for convergence
            if abs(prev_error - current_error) < self.tol:
                self.error = current_error
                self._fitted = True
                return
            
            # Update centroids
            new_centroids = np.array([
                mat[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                else self.centroids[i]
                for i in range(self.k)
            ])
            
            self.centroids = new_centroids
            prev_error = current_error
        
        self.error = prev_error
        self._fitted = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling predict()")
        
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"mat must be a numpy array, got {type(mat)}")
        if mat.ndim != 2:
            raise ValueError(f"mat must be 2D, got shape {mat.shape}")
        if mat.shape[0] == 0:
            raise ValueError(f"mat cannot be empty, got shape {mat.shape}")
        
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"mat has {mat.shape[1]} features but model was fit with "
                f"{self.centroids.shape[1]} features"
            )
        
        distances = cdist(mat, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling get_error()")
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling get_centroids()")
        return self.centroids.copy()

    def _initialize_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using k-means++ algorithm.
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
                
        outputs:
            np.ndarray
                A k x m array of initial centroids
        """
        n_samples = mat.shape[0]
        centroids = []
        
        # Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids.append(mat[first_idx])
        
        # Choose remaining centroids
        for _ in range(1, self.k):
            # Compute distances from each point to nearest centroid
            distances = cdist(mat, np.array(centroids), metric='euclidean')
            min_distances = np.min(distances, axis=1)
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = (min_distances ** 2) / np.sum(min_distances ** 2)
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(mat[next_idx])
        
        return np.array(centroids)
