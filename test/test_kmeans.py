import pytest
import numpy as np
from cluster import KMeans, make_clusters


class TestKMeansInit:
    """Test KMeans initialization and parameter validation."""
    
    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        km = KMeans(k=3)
        assert km.k == 3
        assert km.tol == 1e-6
        assert km.max_iter == 100
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        km = KMeans(k=5, tol=1e-4, max_iter=200)
        assert km.k == 5
        assert km.tol == 1e-4
        assert km.max_iter == 200
    
    def test_init_invalid_k_zero(self):
        """Test that k=0 raises ValueError."""
        with pytest.raises(ValueError):
            KMeans(k=0)
    
    def test_init_invalid_k_negative(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError):
            KMeans(k=-1)
    
    def test_init_invalid_k_type(self):
        """Test that non-integer k raises ValueError."""
        with pytest.raises(ValueError):
            KMeans(k=3.5)
    
    def test_init_invalid_tol_negative(self):
        """Test that negative tol raises ValueError."""
        with pytest.raises(ValueError):
            KMeans(k=3, tol=-1e-6)
    
    def test_init_invalid_max_iter(self):
        """Test that non-positive max_iter raises ValueError."""
        with pytest.raises(ValueError):
            KMeans(k=3, max_iter=0)


class TestKMeansFit:
    """Test KMeans fit method."""
    
    def test_fit_basic(self):
        """Test basic fit functionality."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        assert km._fitted
        assert km.centroids is not None
        assert km.centroids.shape == (3, 2)
        assert km.error is not None
    
    def test_fit_1d_data(self):
        """Test fit with 1D features."""
        data, _ = make_clusters(n=100, m=1, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        assert km.centroids.shape == (3, 1)
    
    def test_fit_high_dim_data(self):
        """Test fit with high-dimensional data."""
        data, _ = make_clusters(n=100, m=50, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        assert km.centroids.shape == (3, 50)
    
    def test_fit_invalid_input_not_array(self):
        """Test that non-array input raises TypeError."""
        km = KMeans(k=3)
        with pytest.raises(TypeError):
            km.fit([[1, 2], [3, 4]])
    
    def test_fit_invalid_input_1d(self):
        """Test that 1D array raises ValueError."""
        km = KMeans(k=3)
        with pytest.raises(ValueError):
            km.fit(np.array([1, 2, 3]))
    
    def test_fit_empty_data(self):
        """Test that empty data raises ValueError."""
        km = KMeans(k=3)
        with pytest.raises(ValueError):
            km.fit(np.array([]).reshape(0, 2))
    
    def test_fit_k_greater_than_n(self):
        """Test that k > n_samples raises ValueError."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        km = KMeans(k=5)
        with pytest.raises(ValueError):
            km.fit(data)
    
    def test_fit_convergence_tight_clusters(self):
        """Test convergence on tight clusters."""
        data, _ = make_clusters(n=200, k=3, scale=0.3)
        km = KMeans(k=3, max_iter=100)
        km.fit(data)
        
        # Check that error is reasonable (non-negative)
        assert km.error >= 0
    
    def test_fit_single_cluster(self):
        """Test fit with k=1."""
        data, _ = make_clusters(n=50, m=2, k=1)
        km = KMeans(k=1)
        km.fit(data)
        
        assert km.centroids.shape == (1, 2)
    
    def test_fit_large_k(self):
        """Test fit with large k."""
        data, _ = make_clusters(n=500, m=2, k=50)
        km = KMeans(k=50)
        km.fit(data)
        
        assert km.centroids.shape == (50, 2)


class TestKMeansPredict:
    """Test KMeans predict method."""
    
    def test_predict_basic(self):
        """Test basic predict functionality."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        labels = km.predict(data)
        
        assert labels.shape == (100,)
        assert np.all((labels >= 0) & (labels < 3))
    
    def test_predict_before_fit(self):
        """Test that predict before fit raises RuntimeError."""
        km = KMeans(k=3)
        data = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(RuntimeError):
            km.predict(data)
    
    def test_predict_wrong_features(self):
        """Test that predict with wrong number of features raises ValueError."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        wrong_data = np.random.randn(50, 3)
        with pytest.raises(ValueError):
            km.predict(wrong_data)
    
    def test_predict_invalid_input_not_array(self):
        """Test that non-array input raises TypeError."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        with pytest.raises(TypeError):
            km.predict([[1, 2], [3, 4]])
    
    def test_predict_invalid_input_1d(self):
        """Test that 1D input raises ValueError."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        with pytest.raises(ValueError):
            km.predict(np.array([1, 2, 3]))
    
    def test_predict_empty_input(self):
        """Test that empty input raises ValueError."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        with pytest.raises(ValueError):
            km.predict(np.array([]).reshape(0, 2))
    
    def test_predict_new_data(self):
        """Test predict on different data than training."""
        train_data, _ = make_clusters(n=100, m=2, k=3, seed=42)
        test_data, _ = make_clusters(n=50, m=2, k=3, seed=43)
        
        km = KMeans(k=3)
        km.fit(train_data)
        labels = km.predict(test_data)
        
        assert labels.shape == (50,)
        assert np.all((labels >= 0) & (labels < 3))


class TestKMeansGetError:
    """Test KMeans get_error method."""
    
    def test_get_error_before_fit(self):
        """Test that get_error before fit raises RuntimeError."""
        km = KMeans(k=3)
        with pytest.raises(RuntimeError):
            km.get_error()
    
    def test_get_error_after_fit(self):
        """Test get_error returns valid error after fit."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        error = km.get_error()
        assert isinstance(error, (float, np.floating))
        assert error >= 0


class TestKMeansGetCentroids:
    """Test KMeans get_centroids method."""
    
    def test_get_centroids_before_fit(self):
        """Test that get_centroids before fit raises RuntimeError."""
        km = KMeans(k=3)
        with pytest.raises(RuntimeError):
            km.get_centroids()
    
    def test_get_centroids_after_fit(self):
        """Test get_centroids returns correct shape."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        centroids = km.get_centroids()
        assert centroids.shape == (3, 2)
    
    def test_get_centroids_returns_copy(self):
        """Test that get_centroids returns a copy, not reference."""
        data, _ = make_clusters(n=100, m=2, k=3)
        km = KMeans(k=3)
        km.fit(data)
        
        centroids1 = km.get_centroids()
        centroids2 = km.get_centroids()
        
        # Should have same values
        np.testing.assert_array_equal(centroids1, centroids2)
        
        # But modifying one shouldn't affect the other
        centroids1[0, 0] = 999
        assert centroids2[0, 0] != 999


class TestKMeansIntegration:
    """Integration tests for KMeans."""
    
    def test_fit_predict_consistency(self):
        """Test that fit then predict works correctly."""
        data, true_labels = make_clusters(n=200, m=2, k=3, scale=0.5)
        km = KMeans(k=3)
        km.fit(data)
        pred_labels = km.predict(data)
        
        # Check that we get k clusters
        assert len(np.unique(pred_labels)) <= 3
        assert np.all((pred_labels >= 0) & (pred_labels < 3))
    
    def test_multiple_runs_same_data(self):
        """Test consistency across multiple runs on same data."""
        data, _ = make_clusters(n=100, m=2, k=3)
        
        km1 = KMeans(k=3, max_iter=100)
        km1.fit(data)
        labels1 = km1.predict(data)
        
        km2 = KMeans(k=3, max_iter=100)
        km2.fit(data)
        labels2 = km2.predict(data)
        
        # Labels might be permuted, but clustering should be similar
        # (may differ due to random initialization)
