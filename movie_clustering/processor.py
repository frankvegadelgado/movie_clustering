import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Set, Any
import yaml
import os
import warnings
warnings.filterwarnings('ignore')


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self.get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        default_config = self.get_default_config()
        config = self.merge_configs(default_config, config)
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'graph': {
                'similarity_threshold': 0.05,
                'min_term_frequency': 2,
                'title_term_weight': 0.3,
                'genome_relevance_threshold': 0.3,
                'max_top_tags': 200,
                'min_title_word_length': 3
            },
            'clustering': {
                'n_clusters': 10,
                'max_iterations': 300,
                'random_state': 42,
                'n_init': 10,
                'top_n_terms_per_cluster': 5
            },
            'data': {
                'movies_path': 'data/raw/ml-25m/movies.csv',
                'tags_path': 'data/raw/ml-25m/tags.csv',
                'genome_scores_path': 'data/raw/ml-25m/genome-scores.csv',
                'genome_tags_path': 'data/raw/ml-25m/genome-tags.csv',
                'output_path': 'data/processed/clusters.csv',
                'visualization_output_path': 'data/processed/visualizations/'
            },
            'processing': {
                'use_tags': False,  # Disabled by default for stability
                'use_genome_tags': False,  # Disabled by default for stability
                'use_title_terms': False,
                'normalize_vectors': True,
                'save_intermediate': False
            }
        }
    
    def merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        def deep_merge(d1, d2):
            for key in d2:
                if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    deep_merge(d1[key], d2[key])
                else:
                    d1[key] = d2[key]
            return d1
        
        result = self.deep_copy(default)
        return deep_merge(result, user)
    
    def deep_copy(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return d
        return {k: self.deep_copy(v) for k, v in d.items()}
    
    def get_graph_config(self) -> Dict[str, Any]:
        return self.config.get('graph', {})
    
    def get_clustering_config(self) -> Dict[str, Any]:
        return self.config.get('clustering', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        return self.config.get('processing', {})
    
    def save_config(self, output_path: str = None):
        save_path = output_path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {save_path}")
    
    def print_config(self):
        print("Current Configuration:")
        print("-" * 50)
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False))


class MovieCatalogGraphProcessor:
    def __init__(self, config: ConfigManager = None, config_path: str = None):
        if config is None:
            if config_path is None:
                config_path = "config.yaml"
            config = ConfigManager(config_path)
        
        self.config = config
        self.graph_config = config.get_graph_config()
        self.clustering_config = config.get_clustering_config()
        self.data_config = config.get_data_config()
        self.processing_config = config.get_processing_config()
        
        self.nlp = None
        self.term_to_idx = {}
        self.idx_to_term = {}
        self.graph = None
        self.vertex_cover = None
        
        print(f"Processor initialized with configuration from {config.config_path}")
        
    def load_movielens_data(self, movies_path: str = None, tags_path: str = None,
                           genome_scores_path: str = None, genome_tags_path: str = None):
        print("Loading MovieLens data...")
        
        movies_path = movies_path or self.data_config.get('movies_path')
        tags_path = tags_path or self.data_config.get('tags_path')
        genome_scores_path = genome_scores_path or self.data_config.get('genome_scores_path')
        genome_tags_path = genome_tags_path or self.data_config.get('genome_tags_path')
        
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        
        movies_df = pd.read_csv(movies_path)
        print(f"Loaded {len(movies_df)} movies")
        
        # Process genres
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            for genre in genres.split('|'):
                if genre.lower() in ['(no genres listed)', '', 'no genres listed']:
                    continue
                all_genres.add(genre.lower())
        
        print(f"Found {len(all_genres)} unique genres: {sorted(all_genres)}")
        terms = list(all_genres)
        
        # Add tags if available
        if tags_path and os.path.exists(tags_path) and self.processing_config.get('use_tags', True):
            print("Loading tags...")
            tags_df = pd.read_csv(tags_path)
            max_top_tags = self.graph_config.get('max_top_tags', 200)
            top_tags = tags_df['tag'].value_counts().head(max_top_tags).index.tolist()
            tag_terms = [str(tag).lower() for tag in top_tags]
            terms.extend(tag_terms)
            print(f"Added {len(tag_terms)} tags")
        
        # Add genome tags if available
        if (genome_scores_path and genome_tags_path and 
            os.path.exists(genome_scores_path) and os.path.exists(genome_tags_path) and
            self.processing_config.get('use_genome_tags', True)):
            
            print("Loading genome tags...")
            genome_tags = pd.read_csv(genome_tags_path)
            genome_scores = pd.read_csv(genome_scores_path)
            
            relevance_threshold = self.graph_config.get('genome_relevance_threshold', 0.3)
            relevant_tags = genome_scores[genome_scores['relevance'] > relevance_threshold]
            tag_ids = relevant_tags['tagId'].unique()
            top_genome_tags = genome_tags[genome_tags['tagId'].isin(tag_ids)]['tag'].tolist()
            
            genome_terms = [str(tag).lower() for tag in top_genome_tags]
            terms.extend(genome_terms)
            print(f"Added {len(genome_terms)} genome tags")
        
        # Deduplicate
        unique_terms = list(set(terms))
        print(f"Unique terms before filtering: {len(unique_terms)}")
        
        # Filter by minimum frequency
        min_freq = self.graph_config.get('min_term_frequency', 0)
        if min_freq > 1:
            print(f"Filtering terms with minimum frequency {min_freq}...")
            from collections import Counter
            
            term_counts = Counter()
            for _, row in movies_df.iterrows():
                if pd.isna(row['genres']):
                    continue
                for genre in str(row['genres']).split('|'):
                    genre_lower = genre.lower()
                    if genre_lower in unique_terms:
                        term_counts[genre_lower] += 1
            
            filtered_terms = [term for term in unique_terms if term_counts.get(term, 0) >= min_freq]
            print(f"Terms after frequency filtering: {len(filtered_terms)}")
            unique_terms = filtered_terms
        
        if len(unique_terms) == 0:
            raise ValueError("No terms found! Check your data and config settings.")
        
        self.term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}
        
        print(f"Final vocabulary: {len(unique_terms)} unique terms")
        
        # Save intermediate data if configured
        if self.processing_config.get('save_intermediate', False):
            self.save_vocabulary(unique_terms)
        
        return movies_df, unique_terms
    
    def save_vocabulary(self, terms: List[str]):
        """Save vocabulary to file."""
        output_dir = os.path.dirname(self.data_config.get('output_path', '.'))
        os.makedirs(output_dir, exist_ok=True)
        
        vocab_path = os.path.join(output_dir, 'vocabulary.txt')
        with open(vocab_path, 'w') as f:
            for term in sorted(terms):
                f.write(f"{term}\n")
        print(f"Vocabulary saved to {vocab_path}")
    
    def build_co_occurrence_matrix(self, movies_df: pd.DataFrame, terms: List[str]):
        print("Building co-occurrence matrix...")
        
        n_terms = len(terms)
        co_occurrence = np.zeros((n_terms, n_terms), dtype=np.float32)
        term_freq = np.zeros(n_terms, dtype=np.float32)
        
        # Single pass through data to get both co-occurrence and frequencies
        for _, row in movies_df.iterrows():
            if pd.isna(row['genres']):
                continue
                
            movie_terms = []
            for genre in str(row['genres']).split('|'):
                genre_lower = genre.lower()
                if genre_lower in self.term_to_idx:
                    term_idx = self.term_to_idx[genre_lower]
                    movie_terms.append(term_idx)
                    term_freq[term_idx] += 1  # Count term frequency
            
            # Update co-occurrences
            movie_terms = list(set(movie_terms))  # Remove duplicates
            for i in range(len(movie_terms)):
                for j in range(i+1, len(movie_terms)):
                    term_i, term_j = movie_terms[i], movie_terms[j]
                    co_occurrence[term_i, term_j] += 1
                    co_occurrence[term_j, term_i] += 1
        
        # Calculate Jaccard similarity
        # Jaccard similarity = |A ∩ B| / |A ∪ B|
        # where |A ∪ B| = |A| + |B| - |A ∩ B|
        similarity_matrix = np.zeros_like(co_occurrence)
        
        for i in range(n_terms):
            for j in range(i+1, n_terms):
                if co_occurrence[i, j] > 0:
                    # intersection = co_occurrence[i, j]
                    # union = freq(i) + freq(j) - intersection
                    union = term_freq[i] + term_freq[j] - co_occurrence[i, j]
                    if union > 0:
                        similarity_matrix[i, j] = co_occurrence[i, j] / union
                        similarity_matrix[j, i] = similarity_matrix[i, j]
        
        print(f"Similarity matrix built: {similarity_matrix.shape}")
        print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        print(f"Non-zero similarities: {np.count_nonzero(similarity_matrix)} out of {n_terms * n_terms}")
        
        # Print some example similarities for debugging
        if n_terms > 0:
            print("\nSample term frequencies:")
            for i in range(min(5, n_terms)):
                print(f"  {self.idx_to_term[i]}: {int(term_freq[i])} movies")
            
            print("\nSample similarities (first few pairs with non-zero similarity):")
            count = 0
            for i in range(n_terms):
                for j in range(i+1, n_terms):
                    if similarity_matrix[i, j] > 0 and count < 5:
                        print(f"  {self.idx_to_term[i]} <-> {self.idx_to_term[j]}: {similarity_matrix[i, j]:.4f}")
                        count += 1
                if count >= 5:
                    break
        
        # Save intermediate data if configured
        if self.processing_config.get('save_intermediate', False):
            self.save_similarity_matrix(similarity_matrix)
        
        return similarity_matrix
    
    def save_similarity_matrix(self, similarity_matrix: np.ndarray):
        """Save similarity matrix to file."""
        output_dir = os.path.dirname(self.data_config.get('output_path', '.'))
        os.makedirs(output_dir, exist_ok=True)
        
        matrix_path = os.path.join(output_dir, 'similarity_matrix.npy')
        np.save(matrix_path, similarity_matrix)
        print(f"Similarity matrix saved to {matrix_path}")
    
    def build_term_graph(self, similarity_matrix: np.ndarray):
        print("Building term graph...")
        
        similarity_threshold = self.graph_config.get('similarity_threshold', 0.05)
        
        self.graph = nx.Graph()
        
        for idx, term in self.idx_to_term.items():
            self.graph.add_node(idx, label=term)
        
        n_terms = similarity_matrix.shape[0]
        edge_count = 0
        
        for i in range(n_terms):
            for j in range(i+1, n_terms):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    self.graph.add_edge(i, j, weight=similarity)
                    edge_count += 1
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {edge_count} edges")
        print(f"Graph density: {nx.density(self.graph):.4f}")
        
        # If no edges, warn and suggest lowering threshold
        if edge_count == 0:
            print(f"\nWARNING: No edges created with threshold {similarity_threshold}")
            print(f"Consider lowering similarity_threshold in config (current: {similarity_threshold})")
            print(f"Suggested: Try 0.01 to 0.1 range")
            
            # Fallback: use all terms if no edges
            print("\nFallback: Using all terms as features since graph has no edges")
        
        # Save intermediate data if configured
        if self.processing_config.get('save_intermediate', False):
            self.save_graph_info()
        
        return self.graph
    
    def save_graph_info(self):
        """Save graph information to file."""
        output_dir = os.path.dirname(self.data_config.get('output_path', '.'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Save graph as GraphML
        graph_path = os.path.join(output_dir, 'term_graph.graphml')
        nx.write_graphml(self.graph, graph_path)
        
        # Save graph statistics
        stats_path = os.path.join(output_dir, 'graph_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Number of nodes: {self.graph.number_of_nodes()}\n")
            f.write(f"Number of edges: {self.graph.number_of_edges()}\n")
            f.write(f"Graph density: {nx.density(self.graph):.6f}\n")
            f.write(f"Average degree: {2 * self.graph.number_of_edges() / self.graph.number_of_nodes():.2f}\n")
            if nx.is_connected(self.graph):
                f.write(f"Diameter: {nx.diameter(self.graph)}\n")
            else:
                f.write(f"Number of connected components: {nx.number_connected_components(self.graph)}\n")
        
        print(f"Graph saved to {graph_path}")
        print(f"Graph statistics saved to {stats_path}")
    
    def find_vertex_cover(self):
        print("Finding approximate vertex cover...")
        
        if self.graph is None:
            raise ValueError("Graph has not been built")
        
        # If graph has no edges, use all nodes
        if self.graph.number_of_edges() == 0:
            print("Graph has no edges - using all nodes as vertex cover")
            self.vertex_cover = set(self.graph.nodes())
        else:
            # Use hvala for vertex cover
            from hvala.algorithm import find_vertex_cover
            self.vertex_cover = find_vertex_cover(self.graph)
        
        print(f"Vertex cover found: {len(self.vertex_cover)} terms")
        if len(self.vertex_cover) > 0:
            sample_terms = [self.idx_to_term[idx] for idx in list(self.vertex_cover)[:10]]
            print("Selected terms (first 10):", sample_terms)
        
        # Save vertex cover if configured
        if self.processing_config.get('save_intermediate', False):
            self.save_vertex_cover()
        
        return self.vertex_cover
    
    def save_vertex_cover(self):
        """Save vertex cover terms to file."""
        output_dir = os.path.dirname(self.data_config.get('output_path', '.'))
        os.makedirs(output_dir, exist_ok=True)
        
        cover_path = os.path.join(output_dir, 'vertex_cover_terms.txt')
        with open(cover_path, 'w') as f:
            for idx in sorted(self.vertex_cover):
                term = self.idx_to_term[idx]
                f.write(f"{term} (index: {idx})\n")
        
        print(f"Vertex cover terms saved to {cover_path}")
    
    def create_movie_vectors(self, movies_df: pd.DataFrame):
        print("Creating movie vectors...")
        
        if self.vertex_cover is None or len(self.vertex_cover) == 0:
            raise ValueError("Vertex cover is empty! Cannot create feature vectors.")
        
        n_movies = len(movies_df)
        n_features = len(self.vertex_cover)
        
        term_to_feature_idx = {term_idx: idx for idx, term_idx in enumerate(self.vertex_cover)}
        
        movie_vectors = np.zeros((n_movies, n_features), dtype=np.float32)
        movie_titles = []
        movie_ids = []
        
        for movie_idx, (_, row) in enumerate(movies_df.iterrows()):
            movie_titles.append(row['title'])
            if 'movieId' in row:
                movie_ids.append(row['movieId'])
            else:
                movie_ids.append(movie_idx)
            
            if pd.isna(row['genres']):
                continue
            
            for genre in str(row['genres']).split('|'):
                genre_lower = genre.lower()
                if genre_lower in self.term_to_idx:
                    term_idx = self.term_to_idx[genre_lower]
                    if term_idx in term_to_feature_idx:
                        feature_idx = term_to_feature_idx[term_idx]
                        movie_vectors[movie_idx, feature_idx] = 1.0
        
        # Normalize
        if self.processing_config.get('normalize_vectors', True):
            norms = np.linalg.norm(movie_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            movie_vectors = movie_vectors / norms
        
        print(f"Vectors created: {movie_vectors.shape}")
        
        # Check how many movies have at least one feature
        non_zero_movies = np.count_nonzero(movie_vectors.sum(axis=1))
        print(f"Movies with at least one feature: {non_zero_movies} / {n_movies}")
        
        # Save vectors if configured
        if self.processing_config.get('save_intermediate', False):
            self.save_movie_vectors(movie_vectors, movie_titles)
        
        return movie_vectors, movie_titles, movie_ids
    
    def save_movie_vectors(self, vectors: np.ndarray, titles: List[str]):
        """Save movie vectors to file."""
        output_dir = os.path.dirname(self.data_config.get('output_path', '.'))
        os.makedirs(output_dir, exist_ok=True)
        
        vectors_path = os.path.join(output_dir, 'movie_vectors.npy')
        np.save(vectors_path, vectors)
        
        titles_path = os.path.join(output_dir, 'movie_titles.txt')
        with open(titles_path, 'w') as f:
            for title in titles:
                f.write(f"{title}\n")
        
        print(f"Movie vectors saved to {vectors_path}")
        print(f"Movie titles saved to {titles_path}")
    
    def spherical_kmeans_clustering(self, movie_vectors: np.ndarray, n_clusters: int = None):
        if n_clusters is None:
            n_clusters = self.clustering_config.get('n_clusters', 10)
        
        print(f"Applying Spherical K-Means with {n_clusters} clusters...")
        
        if movie_vectors.shape[1] == 0:
            raise ValueError("Movie vectors have 0 features! Cannot perform clustering.")
        
        random_state = self.clustering_config.get('random_state', 42)
        max_iter = self.clustering_config.get('max_iterations', 300)
        n_init = self.clustering_config.get('n_init', 10)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter
        )
        cluster_labels = kmeans.fit_predict(movie_vectors)
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(movie_vectors, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(movie_vectors, cluster_labels)
                davies_bouldin = davies_bouldin_score(movie_vectors, cluster_labels)
                
                print(f"Silhouette Score: {silhouette:.3f}")
                print(f"Calinski-Harabasz Index: {calinski_harabasz:.1f}")
                print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
                
                return cluster_labels, {
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin,
                    'n_clusters': n_clusters
                }
        except Exception as e:
            print(f"Could not compute metrics: {e}")
        
        return cluster_labels, None
    
    def analyze_clusters(self, movies_df: pd.DataFrame, cluster_labels: np.ndarray, 
                        movie_titles: List[str], top_n: int = None):
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        
        if top_n is None:
            top_n = self.clustering_config.get('top_n_terms_per_cluster', 5)
        
        results_df = movies_df.copy()
        results_df['cluster'] = cluster_labels
        
        n_clusters = len(set(cluster_labels))
        cluster_stats = {}
        
        for cluster_id in range(n_clusters):
            cluster_movies = results_df[results_df['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id}: {len(cluster_movies)} movies")
            
            term_counts = defaultdict(int)
            for _, row in cluster_movies.iterrows():
                if pd.isna(row['genres']):
                    continue
                for genre in str(row['genres']).split('|'):
                    genre_lower = genre.lower()
                    if genre_lower in self.term_to_idx:
                        term_counts[genre_lower] += 1
            
            top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            print(f"  Top {top_n} terms: {[term for term, _ in top_terms]}")
            
            sample_movies = cluster_movies.head(3)['title'].tolist()
            print(f"  Example movies: {sample_movies}")
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_movies),
                'top_terms': [term for term, _ in top_terms],
                'top_term_counts': [count for _, count in top_terms],
                'example_movies': sample_movies
            }
        
        # Visualization
        self.visualize_clusters(results_df, cluster_stats)
        
        return results_df, cluster_stats
    
    def visualize_clusters(self, results_df: pd.DataFrame, cluster_stats: Dict):
        """
        Visualize cluster distribution and terms.
        
        Args:
            results_df: DataFrame with cluster assignments
            cluster_stats: Dictionary with cluster statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Cluster distribution (bar chart)
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Cluster ID', fontsize=12)
        axes[0, 0].set_ylabel('Number of movies', fontsize=12)
        axes[0, 0].set_title('Movie Distribution by Cluster', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(cluster_counts.values):
            axes[0, 0].text(i, v + max(cluster_counts.values)*0.01, str(v), 
                          ha='center', va='bottom', fontsize=10)
        
        # Top terms per cluster (heatmap)
        all_terms = set()
        for genres in results_df['genres'].dropna():
            for genre in genres.split('|'):
                all_terms.add(genre.lower())
        
        # Take top 15 terms overall
        term_freq = defaultdict(int)
        for genres in results_df['genres'].dropna():
            for genre in genres.split('|'):
                term_freq[genre.lower()] += 1
        
        top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        top_term_names = [term for term, _ in top_terms]
        
        cluster_term_matrix = np.zeros((len(cluster_stats), len(top_term_names)))
        
        for cluster_id in cluster_stats:
            for term_idx, term in enumerate(top_term_names):
                if term in cluster_stats[cluster_id]['top_terms']:
                    term_pos = cluster_stats[cluster_id]['top_terms'].index(term)
                    cluster_term_matrix[cluster_id, term_idx] = cluster_stats[cluster_id]['top_term_counts'][term_pos]
        
        # Normalize by cluster size
        for i in range(len(cluster_stats)):
            if cluster_stats[i]['size'] > 0:
                cluster_term_matrix[i, :] = cluster_term_matrix[i, :] / cluster_stats[i]['size']
        
        sns.heatmap(cluster_term_matrix, ax=axes[0, 1], 
                   xticklabels=top_term_names, yticklabels=range(len(cluster_stats)),
                   cmap='YlOrRd', cbar_kws={'label': 'Normalized Frequency'})
        axes[0, 1].set_xlabel('Term', fontsize=12)
        axes[0, 1].set_ylabel('Cluster', fontsize=12)
        axes[0, 1].set_title('Characteristic Terms per Cluster (Normalized)', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cluster size distribution (pie chart)
        sizes = [stats['size'] for stats in cluster_stats.values()]
        labels = [f'Cluster {i}' for i in cluster_stats.keys()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].axis('equal')
        
        # Term importance in vertex cover
        if self.vertex_cover is not None:
            cover_terms = [self.idx_to_term[idx] for idx in self.vertex_cover]
            cover_term_freq = {term: term_freq.get(term, 0) for term in cover_terms}
            
            # Take top 10 vertex cover terms
            top_cover_terms = sorted(cover_term_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            terms_list = [term for term, _ in top_cover_terms]
            freq_list = [freq for _, freq in top_cover_terms]
            
            y_pos = np.arange(len(terms_list))
            axes[1, 1].barh(y_pos, freq_list, color='lightcoral', edgecolor='black')
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(terms_list)
            axes[1, 1].set_xlabel('Frequency in Dataset', fontsize=12)
            axes[1, 1].set_title('Top 10 Vertex Cover Terms by Frequency', fontsize=14, fontweight='bold')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save visualization
        vis_dir = self.data_config.get('visualization_output_path', 'data/processed/visualizations/')
        os.makedirs(vis_dir, exist_ok=True)
        
        vis_path = os.path.join(vis_dir, 'cluster_analysis.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {vis_path}")
        
        plt.show()
         
    def save_results(self, results_df: pd.DataFrame, metrics: Dict[str, Any] = None):
        output_path = self.data_config.get('output_path', 'data/processed/clusters.csv')
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        print(f"Clustering results saved to {output_path}")
        
        if metrics:
            metrics_path = os.path.join(output_dir, 'clusters_metrics.txt')
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"Clustering metrics saved to {metrics_path}")

def main(config_path: str = "config.yaml"):
    print("="*60)
    print("MOVIE CATALOG CLUSTERING PIPELINE")
    print("="*60)
    
    config = ConfigManager(config_path)
    config.print_config()
    
    processor = MovieCatalogGraphProcessor(config)
    
    data_config = config.get_data_config()
    movies_df, terms = processor.load_movielens_data(
        movies_path=data_config.get('movies_path'),
        tags_path=data_config.get('tags_path'),
        genome_scores_path=data_config.get('genome_scores_path'),
        genome_tags_path=data_config.get('genome_tags_path')
    )
    
    similarity_matrix = processor.build_co_occurrence_matrix(movies_df, terms)
    graph = processor.build_term_graph(similarity_matrix)
    vertex_cover = processor.find_vertex_cover()
    
    movie_vectors, movie_titles, movie_ids = processor.create_movie_vectors(movies_df)
    
    n_clusters = config.get_clustering_config().get('n_clusters', 10)
    cluster_labels, metrics = processor.spherical_kmeans_clustering(movie_vectors, n_clusters)
    
    top_n = config.get_clustering_config().get('top_n_terms_per_cluster', 5)
    results_df, cluster_stats = processor.analyze_clusters(movies_df, cluster_labels, movie_titles, top_n)
    
    processor.save_results(results_df, metrics)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    if not os.path.exists("config.yaml"):
        print("Creating default config.yaml file...")
        config_manager = ConfigManager()
        config_manager.save_config()
    
    main()