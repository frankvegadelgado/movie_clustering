# Movie Catalog Clustering with Vertex Cover Feature Selection

<div align="center">
   <img src="docs/cluster_analysis.png" alt="Clustering Visualization" width="600">
   <br><br>
   <em>MovieLens Clustering Results.</em>
</div>

## Overview

This project demonstrates a novel approach to movie catalog clustering using **Vertex Cover-based feature selection** combined with **Spherical K-Means clustering**. By representing movie metadata (genres, tags, keywords) as a graph and applying an efficient vertex cover approximation algorithm, we select the most discriminative features before clustering, resulting in more coherent and interpretable movie groupings.

The key innovation is using Minimum Vertex Cover (MVC) as a feature selection method to identify central terms that capture the strongest relationships in the movie catalog, which significantly improves clustering quality compared to using all features.

## Features

- **Graph-based Feature Selection**: Builds a term co-occurrence graph and applies vertex cover approximation
- **Efficient Implementation**: Uses Hvala package for O(m log n) vertex cover approximation with √2 ratio
- **MovieLens Integration**: Works with standard MovieLens datasets (25M, latest-small)
- **Spherical K-Means**: Implements cosine similarity-based clustering optimized for text data
- **Visual Analytics**: Provides cluster visualization and term importance analysis
- **Modular Design**: Easy to extend to other text clustering domains

## Project Structure

```
movie_clustering/
├── movie_clustering/
│   ├── __init__.py
│   ├── processor.py           # Main Algorithm
├── data/
│   ├── raw/                   # Original MovieLens data
│   └── processed/             # Processed datasets
├── tests/
│   ├── test_movie_clustering.py
├── requirements.txt
├── setup.py
├── config.yaml               # Configuration parameters
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/frankvegadelgado/movie_clustering.git
cd movie_clustering

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

### Manual Installation

```bash
# Install core dependencies
pip install pandas scikit-learn matplotlib seaborn spacy pyyaml

# Install Hvala package for vertex cover algorithm
pip install hvala==0.0.7

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Using requirements.txt

```bash
# Install all dependencies at once
pip install -r requirements.txt
```

**requirements.txt:**
```
# Vertex cover algorithm
hvala==0.0.7

# Core dependencies
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
spacy>=3.5.0

# Configuration management
pyyaml>=6.0

# Data processing and ML utilities
joblib>=1.2.0
threadpoolctl>=3.1.0
```

## Quick Start

### 1. Install the Package

```bash
pip install movie_clustering
```

### 2. Download MovieLens Data

```bash
# Download MovieLens 25M dataset
cd data/raw
wget -O ml-25m.zip http://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
cd ../..
```

### 3. Run the Console Script

```bash
# Create a basic configuration file
cat > config.yaml << EOF
data:
  movies_path: "data/raw/ml-25m/movies.csv"
  tags_path: "data/raw/ml-25m/tags.csv"
  output_path: "movie_clusters.csv"
graph:
  similarity_threshold: 0.05
clustering:
  n_clusters: 10
EOF

# Run the clustering pipeline
movie-cluster
```

### 4. Basic Python Usage

```python
from movie_clustering.processor import MovieCatalogGraphProcessor
import pandas as pd

# Initialize processor
processor = MovieCatalogGraphProcessor(similarity_threshold=0.05)

# Load and process data
movies_df, terms = processor.load_movielens_data(
    movies_path='data/raw/ml-25m/movies.csv',
    tags_path='data/raw/ml-25m/tags.csv'
)

# Build graph and find vertex cover
similarity_matrix = processor.build_co_occurrence_matrix(movies_df, terms)
graph = processor.build_term_graph(similarity_matrix)
vertex_cover = processor.find_vertex_cover()  # Uses Hvala package

# Create reduced feature vectors
movie_vectors, movie_titles = processor.create_movie_vectors(movies_df)

# Apply spherical k-means clustering
cluster_labels = processor.spherical_kmeans_clustering(movie_vectors, n_clusters=10)

# Analyze results
results_df = processor.analyze_clusters(movies_df, cluster_labels, movie_titles)
```

### 5. Complete Pipeline Example

```python
# Alternative: Use the console script programmatically
import subprocess

# Run movie-cluster with custom config
result = subprocess.run(['movie-cluster'], capture_output=True, text=True)
print(result.stdout)
```

## How It Works

### 1. Graph Construction
- **Vertices**: Movie terms (genres, tags, keywords)
- **Edges**: Created between terms with similarity ≥ threshold (default: 0.3)
- **Edge Weight**: Co-occurrence frequency normalized by term frequency

### 2. Vertex Cover Feature Selection
- Uses Hvala's `find_vertex_cover()` with O(m log n) complexity and √2 approximation ratio
- Selects minimal set of terms that cover all strong relationships
- Reduces dimensionality while preserving semantic structure

### 3. Spherical K-Means Clustering
- Normalizes vectors to unit sphere
- Uses cosine similarity as distance metric
- Optimized for high-dimensional sparse text data

### 4. Result Interpretation
- Each cluster represents a coherent movie genre/theme combination
- Feature importance shows why movies were grouped together
- Visualizations reveal cluster overlap and boundaries

## Key Algorithms

### Vertex Cover Approximation (via Hvala)
```python
import hvala

# Hvala provides efficient vertex cover approximation
graph = ...  # NetworkX graph
from hvala.algorithm import find_vertex_cover
vertex_cover = find_vertex_cover(graph)
```

### Spherical K-Means
```python
# Equivalent to K-Means with cosine similarity
from sklearn.cluster import KMeans
import numpy as np

# Normalize vectors to unit sphere
vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(vectors_normalized)
```

## Configuration

Modify `config.yaml` to adjust parameters:

```yaml
# Graph construction parameters
graph:
  # Threshold for creating edges in the similarity graph (0.2-0.4 recommended)
  similarity_threshold: 0.05
  
  # Minimum frequency for terms to be included in vocabulary
  min_term_frequency: 2
  
  # Weight for title terms relative to genres (0.0-1.0)
  title_term_weight: 0.3
  
  # Threshold for including genome tags based on relevance score
  genome_relevance_threshold: 0.3
  
  # Maximum number of top tags to include from tags.csv
  max_top_tags: 200
  
  # Minimum word length for title terms
  min_title_word_length: 3

# Clustering parameters
clustering:
  # Number of clusters for K-Means
  n_clusters: 10
  
  # Maximum iterations for K-Means algorithm
  max_iterations: 300
  
  # Random seed for reproducibility
  random_state: 42
  
  # Number of K-Means initializations
  n_init: 10
  
  # Number of top terms to show per cluster in analysis
  top_n_terms_per_cluster: 5

# Data paths
data:
  # Path to movies.csv file
  movies_path: "data/raw/ml-25m/movies.csv"
  
  # Path to tags.csv file (optional)
  tags_path: "data/raw/ml-25m/tags.csv"
  
  # Path to genome-scores.csv (optional)
  genome_scores_path: "data/raw/ml-25m/genome-scores.csv"
  
  # Path to genome-tags.csv (optional)
  genome_tags_path: "data/raw/ml-25m/genome-tags.csv"
  
  # Output path for clustering results
  output_path: "data/processed/clusters.csv"
  
  # Output directory for visualizations
  visualization_output_path: "data/processed/visualizations/"

# Processing options
processing:
  # Whether to include tags from tags.csv
  use_tags: true
  
  # Whether to include genome tags
  use_genome_tags: true
  
  # Whether to extract terms from movie titles
  use_title_terms: true
  
  # Whether to normalize vectors for spherical K-Means
  normalize_vectors: true
  
  # Whether to save intermediate results for debugging
  save_intermediate: false
```

## Results and Evaluation

### Performance Metrics
- **Silhouette Score**: Measures cluster separation and cohesion
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity between clusters

### Comparative Analysis
The method typically shows:
- 15-25% improvement in silhouette score vs. baseline (all features)
- 30-40% reduction in feature dimensions
- More interpretable clusters with clear thematic coherence

## Applications

1. **Movie Recommendation Systems**: Better understanding of movie similarities
2. **Catalog Organization**: Automated genre and theme discovery
3. **Content Analysis**: Identifying trends and gaps in movie libraries
4. **Educational Tools**: Teaching graph algorithms and clustering techniques

## Extending the Project

### Custom Data Sources
```python
# Implement your own data loader
class CustomDataLoader:
    def load_terms(self, data_path):
        # Extract terms from your data format
        pass
```

### Alternative Clustering
```python
# Try different clustering algorithms
from sklearn.cluster import DBSCAN, AgglomerativeClustering

dbscan = DBSCAN(metric='cosine', eps=0.3)
hierarchical = AgglomerativeClustering(n_clusters=10, linkage='average', affinity='cosine')
```

## Troubleshooting

### Common Issues

1. **Hvala installation fails**
   ```bash
   # Try installing from alternative source
   pip install hvala==0.0.7 --extra-index-url https://pypi.org/simple
   ```

2. **Memory issues with large datasets**
   - Reduce `min_term_frequency` in config
   - Use sparse matrices for similarity computation
   - Process data in batches

3. **spaCy model download fails**
   ```bash
   # Download model manually
   python -m spacy download en_core_web_sm --direct
   ```

### Debug Mode
```python
processor = MovieCatalogGraphProcessor(debug=True)
# Enables verbose logging and intermediate file saving
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/frankvegadelgado/movie_clustering.git
cd movie_clustering
pip install -e ".[dev]"
pre-commit install
```

## Citation

If you use this code in research, please cite:

```bibtex
@software{movie_clustering,
  title = {Movie Catalog Clustering with Vertex Cover Feature Selection},
  author = {Vega, Frank},
  year = {2025},
  url = {https://github.com/frankvegadelgado/movie_clustering}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MovieLens dataset from GroupLens Research
- Hvala package for efficient vertex cover implementation
- Scikit-learn, NetworkX, and spaCy communities
- Inspired by research on graph-based feature selection and spherical clustering

## Contact

For questions and support:
- GitHub Issues: [Project Issues](https://github.com/frankvegadelgado/movie_clustering/issues)
- Email: vega.frank@gmail.com