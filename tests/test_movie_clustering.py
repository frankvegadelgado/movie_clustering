"""
Test suite for movie catalog clustering with vertex cover feature selection.
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add the movie_clustering directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "movie_clustering"))

from movie_clustering.processor import MovieCatalogGraphProcessor, ConfigManager


class TestConfigManager:
    """Test configuration management functionality."""
    
    def test_default_config_creation(self):
        """Test that default configuration is created correctly."""
        config = ConfigManager()
        default_config = config.get_default_config()
        
        # Check main sections exist
        assert 'graph' in default_config
        assert 'clustering' in default_config
        assert 'data' in default_config
        assert 'processing' in default_config
        
        # Check some specific values
        assert default_config['graph']['similarity_threshold'] == 0.05
        assert default_config['clustering']['n_clusters'] == 10
    
    def test_config_loading(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Create a test config file
        test_config = {
            'graph': {'similarity_threshold': 0.05},
            'clustering': {'n_clusters': 5}
        }
        
        config_file = tmp_path / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        config = ConfigManager(str(config_file))
        
        # Test that loaded values override defaults
        assert config.config['graph']['similarity_threshold'] == 0.05
        assert config.config['clustering']['n_clusters'] == 5
    
    def test_config_merging(self):
        """Test merging user config with defaults."""
        config = ConfigManager()
        user_config = {
            'graph': {'similarity_threshold': 0.05}
        }
        
        merged = config.merge_configs(config.get_default_config(), user_config)
        
        # User value should override
        assert merged['graph']['similarity_threshold'] == 0.05
        # Default value should remain
        assert merged['clustering']['n_clusters'] == 10
    
    def test_missing_config_file(self, tmp_path):
        """Test behavior when config file doesn't exist."""
        non_existent = tmp_path / 'non_existent.yaml'
        config = ConfigManager(str(non_existent))
        
        # Should fall back to defaults
        assert config.config is not None
        assert 'graph' in config.config


class TestMovieCatalogGraphProcessor:
    """Test movie catalog graph processor functionality."""
    
    @pytest.fixture
    def synthetic_movies(self):
        """Create synthetic movie data for testing."""
        return pd.DataFrame({
            'movieId': range(10),
            'title': [
                'The Matrix', 'Inception', 'Star Wars', 'Lord of the Rings',
                'The Godfather', 'Pulp Fiction', 'Toy Story', 'Finding Nemo',
                'The Dark Knight', 'Interstellar'
            ],
            'genres': [
                'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 
                'Action|Adventure|Sci-Fi|Fantasy', 'Adventure|Fantasy', 
                'Crime|Drama', 'Crime|Drama', 'Animation|Adventure|Comedy',
                'Animation|Adventure|Comedy', 'Action|Crime|Drama|Thriller',
                'Adventure|Drama|Sci-Fi'
            ]
        })
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create a test configuration for unit tests."""
        config = {
            'graph': {
                'similarity_threshold': 0.05,
                'min_term_frequency': 2,
                'title_term_weight': 0.3,
                'genome_relevance_threshold': 0.3,
                'max_top_tags': 200,
                'min_title_word_length': 3
            },
            'clustering': {
                'n_clusters': 3,
                'max_iterations': 100,
                'random_state': 42,
                'n_init': 5,
                'top_n_terms_per_cluster': 3
            },
            'data': {
                'movies_path': 'test_data/movies.csv',
                'output_path': 'test_output/clusters.csv',
                'visualization_output_path': 'test_output/visualizations/'
            },
            'processing': {
                'use_tags': False,
                'use_genome_tags': False,
                'use_title_terms': False,  # Disable for testing to avoid spaCy dependency
                'normalize_vectors': True,
                'save_intermediate': False
            }
        }
        
        config_file = tmp_path / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_file)
    
    def test_processor_initialization(self, test_config):
        """Test that processor initializes correctly with config."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        assert processor is not None
        assert processor.graph_config is not None
        assert processor.clustering_config is not None
        assert processor.graph_config['similarity_threshold'] == 0.05
        assert processor.clustering_config['n_clusters'] == 3
    
    def test_vocabulary_creation(self, synthetic_movies, test_config):
        """Test vocabulary creation from movie data."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary manually
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Verify vocabulary
        assert len(processor.term_to_idx) == len(terms)
        assert len(processor.idx_to_term) == len(terms)
        
        # Test reverse mapping
        for term, idx in processor.term_to_idx.items():
            assert processor.idx_to_term[idx] == term
    
    def test_co_occurrence_matrix(self, synthetic_movies, test_config):
        """Test co-occurrence matrix construction."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Build co-occurrence matrix
        similarity_matrix = processor.build_co_occurrence_matrix(synthetic_movies, terms)
        
        # Check matrix properties
        n_terms = len(terms)
        assert similarity_matrix.shape == (n_terms, n_terms)
        assert np.all(similarity_matrix >= 0)  # All values should be non-negative
        assert np.all(similarity_matrix <= 1)  # All values should be <= 1 (normalized)
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # Should be symmetric
    
    def test_graph_construction(self, synthetic_movies, test_config):
        """Test graph construction from similarity matrix."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary and similarity matrix
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        similarity_matrix = processor.build_co_occurrence_matrix(synthetic_movies, terms)
        
        # Build graph
        graph = processor.build_term_graph(similarity_matrix)
        
        # Check graph properties
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == len(terms)
        
        # Check edges are created for similarity >= threshold
        threshold = processor.graph_config['similarity_threshold']
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    assert graph.has_edge(i, j)
                    assert graph[i][j]['weight'] == similarity
    
    def test_vertex_cover_approximation(self, synthetic_movies, test_config):
        """Test vertex cover approximation (requires hvala package)."""
        try:
            import hvala
            hvala_available = True
        except ImportError:
            hvala_available = False
            pytest.skip("hvala package not available")
        
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary and graph
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        similarity_matrix = processor.build_co_occurrence_matrix(synthetic_movies, terms)
        graph = processor.build_term_graph(similarity_matrix)
        
        # Find vertex cover
        vertex_cover = processor.find_vertex_cover()
        
        # Check vertex cover properties
        assert vertex_cover is not None
        assert len(vertex_cover) > 0
        assert len(vertex_cover) <= graph.number_of_nodes()
        
        # Check that vertex cover actually covers all edges
        for u, v in graph.edges():
            assert u in vertex_cover or v in vertex_cover
    
    def test_movie_vector_creation(self, synthetic_movies, test_config):
        """Test creation of movie vectors from vertex cover terms."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Create a mock vertex cover (use all terms for testing)
        processor.vertex_cover = set(processor.term_to_idx.values())
        
        # Create movie vectors
        movie_vectors, movie_titles, movie_ids = processor.create_movie_vectors(synthetic_movies)
        
        # Check vector properties
        n_movies = len(synthetic_movies)
        n_features = len(processor.vertex_cover)
        
        assert movie_vectors.shape == (n_movies, n_features)
        assert len(movie_titles) == n_movies
        assert len(movie_ids) == n_movies
        
        # Check vector values (should be binary for genres)
        for i in range(n_movies):
            # Vectors should be normalized if normalize_vectors is True
            if processor.processing_config.get('normalize_vectors', True):
                norm = np.linalg.norm(movie_vectors[i])
                assert np.isclose(norm, 1.0) or np.isclose(norm, 0.0)
    
    def test_clustering(self, synthetic_movies, test_config):
        """Test spherical k-means clustering."""
        processor = MovieCatalogGraphProcessor(config_path=test_config)
        
        # Create vocabulary and vertex cover
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Create a mock vertex cover
        processor.vertex_cover = set(processor.term_to_idx.values())
        
        # Create movie vectors
        movie_vectors, movie_titles, movie_ids = processor.create_movie_vectors(synthetic_movies)
        
        # Apply clustering
        n_clusters = processor.clustering_config['n_clusters']
        cluster_labels, metrics = processor.spherical_kmeans_clustering(
            movie_vectors, n_clusters=n_clusters
        )
        
        # Check clustering results
        assert cluster_labels is not None
        assert len(cluster_labels) == len(synthetic_movies)
        
        # All cluster labels should be between 0 and n_clusters-1
        assert set(cluster_labels).issubset(set(range(n_clusters)))
        
        # Should have at least 1 cluster
        assert len(set(cluster_labels)) >= 1
        
        # Check metrics if computed
        if metrics is not None:
            assert 'silhouette' in metrics
            assert 'n_clusters' in metrics
    
    def test_complete_pipeline(self, synthetic_movies, test_config, tmp_path):
        """Test the complete pipeline with synthetic data (integration test)."""
        # Modify config to use test output paths
        with open(test_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update output paths to use temp directory
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config['data']['output_path'] = str(output_dir / 'clusters.csv')
        config['data']['visualization_output_path'] = str(output_dir / 'visualizations/')
        config['processing']['save_intermediate'] = True
        
        # Save updated config
        updated_config = tmp_path / 'updated_config.yaml'
        with open(updated_config, 'w') as f:
            yaml.dump(config, f)
        
        # Run the pipeline
        processor = MovieCatalogGraphProcessor(config_path=str(updated_config))
        
        # Create vocabulary manually
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Build similarity matrix
        similarity_matrix = processor.build_co_occurrence_matrix(synthetic_movies, terms)
        
        # Build graph
        graph = processor.build_term_graph(similarity_matrix)
        
        # Find vertex cover (skip if hvala not available)
        try:
            import hvala
            vertex_cover = processor.find_vertex_cover()
        except ImportError:
            # Use a simple approximation for testing
            processor.vertex_cover = set(list(processor.term_to_idx.values())[:5])
            vertex_cover = processor.vertex_cover
        
        # Create movie vectors
        movie_vectors, movie_titles, movie_ids = processor.create_movie_vectors(synthetic_movies)
        
        # Apply clustering
        cluster_labels, metrics = processor.spherical_kmeans_clustering(movie_vectors)
        
        # Analyze clusters
        results_df, cluster_stats = processor.analyze_clusters(
            synthetic_movies, cluster_labels, movie_titles
        )
        
        # Save results
        processor.save_results(results_df, metrics)
        
        # Verify outputs were created
        output_path = config['data']['output_path']
        assert os.path.exists(output_path)
        
        # Load and verify saved results
        saved_results = pd.read_csv(output_path)
        assert len(saved_results) == len(synthetic_movies)
        assert 'cluster' in saved_results.columns


def test_quick_test_integration():
    """
    Integration test that mimics the original quick_test function.
    This is kept for backward compatibility.
    """
    # Create synthetic data
    synthetic_movies = pd.DataFrame({
        'movieId': range(10),
        'title': [
            'The Matrix', 'Inception', 'Star Wars', 'Lord of the Rings',
            'The Godfather', 'Pulp Fiction', 'Toy Story', 'Finding Nemo',
            'The Dark Knight', 'Interstellar'
        ],
        'genres': [
            'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 
            'Action|Adventure|Sci-Fi|Fantasy', 'Adventure|Fantasy', 
            'Crime|Drama', 'Crime|Drama', 'Animation|Adventure|Comedy',
            'Animation|Adventure|Comedy', 'Action|Crime|Drama|Thriller',
            'Adventure|Drama|Sci-Fi'
        ]
    })
    
    # Create a temporary configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'graph': {
                'similarity_threshold': 0.05,
                'min_term_frequency': 2,
                'use_title_terms': False
            },
            'clustering': {
                'n_clusters': 3
            },
            'processing': {
                'use_title_terms': False
            }
        }
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Run the test
        processor = MovieCatalogGraphProcessor(config_path=config_path)
        
        # Create vocabulary manually
        all_terms = set()
        for genres in synthetic_movies['genres']:
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        terms = list(all_terms)
        processor.term_to_idx = {term: idx for idx, term in enumerate(terms)}
        processor.idx_to_term = {idx: term for term, idx in processor.term_to_idx.items()}
        
        # Build similarity matrix
        similarity_matrix = processor.build_co_occurrence_matrix(synthetic_movies, terms)
        
        # Build graph
        graph = processor.build_term_graph(similarity_matrix)
        
        # Mock vertex cover (since we might not have hvala in test environment)
        processor.vertex_cover = set(range(min(5, len(terms))))
        
        # Create movie vectors
        movie_vectors, movie_titles, movie_ids = processor.create_movie_vectors(synthetic_movies)
        
        # Apply clustering
        cluster_labels, metrics = processor.spherical_kmeans_clustering(movie_vectors, n_clusters=3)
        
        # Get results
        results_df = synthetic_movies.copy()
        results_df['cluster'] = cluster_labels
        
        # Verify results
        assert 'cluster' in results_df.columns
        assert len(results_df) == 10
        assert set(results_df['cluster'].unique()).issubset({0, 1, 2})
        
        print("\n=== CLUSTERING RESULTS ===")
        for cluster_id in sorted(results_df['cluster'].unique()):
            cluster_movies = results_df[results_df['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id}:")
            for _, row in cluster_movies.iterrows():
                print(f"  - {row['title']} ({row['genres']})")
        
    finally:
        # Clean up
        os.unlink(config_path)


if __name__ == "__main__":
    """
    Run the tests directly for debugging.
    """
    # Create a simple test
    test_quick_test_integration()
    print("\nAll tests passed!")