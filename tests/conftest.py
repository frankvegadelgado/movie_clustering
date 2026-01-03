"""
Pytest configuration file for movie clustering tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def sample_movie_data():
    """Provide sample movie data for multiple tests."""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': [
            'Action|Adventure',
            'Comedy|Romance',
            'Drama|Thriller',
            'Action|Thriller',
            'Comedy|Drama'
        ]
    })


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = {
        'graph': {
            'similarity_threshold': 0.05,
            'min_term_frequency': 2,
            'title_term_weight': 0.3
        },
        'clustering': {
            'n_clusters': 3,
            'random_state': 42
        },
        'data': {
            'movies_path': 'test_data.csv',
            'output_path': 'test_output.csv'
        },
        'processing': {
            'use_title_terms': False,
            'normalize_vectors': True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)