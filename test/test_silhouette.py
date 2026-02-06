import pytest
import numpy as np
from cluster import Silhouette, make_clusters
from sklearn.metrics import silhouette_score, silhouette_samples


class TestSilhouetteInit:
    """Test Silhouette initialization."""
    
    def test_init(self):
        """Test basic initialization."""
        sil = Silhouette()
        assert isinstance(sil, Silhouette)


class TestSilhouetteScore:
    """Test Silhouette score method."""
    
    def test_score_basic(self):
        """Test basic score functionality."""
        X, y = make_clusters(n=100, m=2, k=3)
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (100,)
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    
    def test_score_invalid_X_not_array(self):
        """Test that non-array X raises TypeError."""
        sil = Silhouette()
        y = np.array([0, 1, 2])
        
        with pytest.raises(TypeError):
            sil.score([[1, 2], [3, 4]], y)
    
    def test_score_invalid_y_not_array(self):
        """Test that non-array y raises TypeError."""
        sil = Silhouette()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        with pytest.raises(TypeError):
            sil.score(X, [0, 1, 2])
    
    def test_score_invalid_X_1d(self):
        """Test that 1D X raises ValueError."""
        sil = Silhouette()
        y = np.array([0, 1, 2])
        
        with pytest.raises(ValueError):
            sil.score(np.array([1, 2, 3]), y)
    
    def test_score_invalid_y_2d(self):
        """Test that 2D y raises ValueError."""
        sil = Silhouette()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        with pytest.raises(ValueError):
            sil.score(X, np.array([[0], [1], [2]]))
    
    def test_score_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise ValueError."""
        sil = Silhouette()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])
        
        with pytest.raises(ValueError):
            sil.score(X, y)
    
    def test_score_empty_data(self):
        """Test that empty data raises ValueError."""
        sil = Silhouette()
        with pytest.raises(ValueError):
            sil.score(np.array([]).reshape(0, 2), np.array([]))
    
    def test_score_single_cluster(self):
        """Test score with single cluster returns zeros."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (3,)
        np.testing.assert_array_equal(scores, np.zeros(3))
    
    def test_score_two_clusters(self):
        """Test score with two well-separated clusters."""
        # Create two well-separated clusters
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([10, 10])
        ])
        y = np.hstack([np.zeros(50), np.ones(50)]).astype(int)
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (100,)
        # For well-separated clusters, most scores should be positive
        assert np.mean(scores) > 0.5
    
    def test_score_1d_features(self):
        """Test score with 1D features."""
        X = np.array([[1], [2], [3], [10], [11], [12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (6,)
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    
    def test_score_high_dim_features(self):
        """Test score with high-dimensional features."""
        X, y = make_clusters(n=100, m=50, k=3)
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (100,)
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


class TestSilhouetteComparisonWithSklearn:
    """Test Silhouette implementation against sklearn."""
    
    def test_score_matches_sklearn_basic(self):
        """Test that our implementation matches sklearn on basic data."""
        X, y = make_clusters(n=100, m=2, k=3)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        sklearn_scores = silhouette_samples(X, y)
        
        # Check shape
        assert our_scores.shape == sklearn_scores.shape
        
        # Check that scores are reasonably close
        # Allow some tolerance for numerical differences
        np.testing.assert_allclose(our_scores, sklearn_scores, atol=1e-5)
    
    def test_score_mean_matches_sklearn(self):
        """Test that mean score matches sklearn."""
        X, y = make_clusters(n=150, m=2, k=4)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        our_mean = np.mean(our_scores)
        
        sklearn_mean = silhouette_score(X, y)
        
        # Check that mean scores are very close
        np.testing.assert_allclose(our_mean, sklearn_mean, atol=1e-5)
    
    def test_score_matches_sklearn_1d(self):
        """Test matching sklearn on 1D data."""
        X = np.random.randn(50, 1) * 10
        y = np.hstack([np.zeros(25), np.ones(25)]).astype(int)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        sklearn_scores = silhouette_samples(X, y)
        
        np.testing.assert_allclose(our_scores, sklearn_scores, atol=1e-5)
    
    def test_score_matches_sklearn_many_clusters(self):
        """Test matching sklearn with many clusters."""
        X, y = make_clusters(n=200, m=2, k=10)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        sklearn_scores = silhouette_samples(X, y)
        
        np.testing.assert_allclose(our_scores, sklearn_scores, atol=1e-5)
    
    def test_score_matches_sklearn_tight_clusters(self):
        """Test matching sklearn on tight clusters."""
        X, y = make_clusters(n=100, m=2, k=3, scale=0.3)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        sklearn_scores = silhouette_samples(X, y)
        
        np.testing.assert_allclose(our_scores, sklearn_scores, atol=1e-5)
    
    def test_score_matches_sklearn_loose_clusters(self):
        """Test matching sklearn on loose clusters."""
        X, y = make_clusters(n=100, m=2, k=3, scale=2.0)
        
        sil = Silhouette()
        our_scores = sil.score(X, y)
        sklearn_scores = silhouette_samples(X, y)
        
        np.testing.assert_allclose(our_scores, sklearn_scores, atol=1e-5)


class TestSilhouetteEdgeCases:
    """Test edge cases for Silhouette scoring."""
    
    def test_score_single_point_per_cluster(self):
        """Test with one point per cluster."""
        X = np.array([[0, 0], [5, 5], [10, 10]])
        y = np.array([0, 1, 2])
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (3,)
        # Single points per cluster should have well-defined silhouette
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    
    def test_score_identical_points(self):
        """Test with identical points."""
        X = np.array([[1, 1], [1, 1], [1, 1]])
        y = np.array([0, 0, 1])
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (3,)
        # Should not raise an error and scores should be valid
        assert np.all(np.isfinite(scores))
    
    def test_score_very_close_points(self):
        """Test with very close points."""
        X = np.array([
            [0, 0], [0.001, 0.001], [0.002, 0.002],
            [10, 10], [10.001, 10.001], [10.002, 10.002]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        sil = Silhouette()
        scores = sil.score(X, y)
        
        assert scores.shape == (6,)
        assert np.all(np.isfinite(scores))
