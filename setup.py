from setuptools import setup, find_packages

# Core dependencies for the main package
INSTALL_REQUIRES = [
    # Vertex cover algorithm
    "hvala>=0.0.7",
    
    # Core data processing
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    
    # Machine learning
    "scikit-learn>=1.2.0",
    "joblib>=1.2.0",
    "threadpoolctl>=3.1.0",
    
    # Graph processing
    "networkx>=3.6.1",
    
    # Configuration management
    "pyyaml>=6.0.0",
    
    # Scientific computing
    "scipy>=1.10.0",
    
    # NLP processing
    "spacy>=3.5.0",
]

# Optional dependencies for visualization
EXTRAS_REQUIRE = {
    "visualization": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "ipykernel>=6.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "nbsphinx>=0.9.0",
    ],
    "dev": [
        # Testing framework
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.0.0",
        
        # Test utilities
        "hypothesis>=6.0.0",
        "Faker>=18.0.0",
        
        # Mocking
        "pytest-mock>=3.0.0",
        
        # Code quality
        "pytest-flake8>=1.1.0",
        "pytest-black>=0.3.0",
        "pytest-isort>=3.0.0",
        
        # Development tools
        "black>=22.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
        
        # Type stubs for better IDE support
        "types-PyYAML>=6.0.0",
        "types-requests>=2.0.0",
    ],
    "full": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "wordcloud>=1.9.0",
        "tqdm>=4.0.0",
        "nltk>=3.8.0",
        "gensim>=4.3.0",
    ]
}

# Combine all extras
EXTRAS_REQUIRE["all"] = (
    EXTRAS_REQUIRE["visualization"] +
    EXTRAS_REQUIRE["jupyter"] +
    EXTRAS_REQUIRE["docs"] +
    EXTRAS_REQUIRE["dev"] +
    EXTRAS_REQUIRE["full"]
)

setup(
    name="movie_clustering",
    version="0.0.1",
    author="Frank Vega",
    author_email="vega.frank@gmail.com",
    description="Movie catalog clustering using vertex cover feature selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frankvegadelgado/movie_clustering",
    packages=["movie_clustering"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "movie-cluster=movie_clustering.processor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="clustering, vertex-cover, feature-selection, movie-recommendation",
)