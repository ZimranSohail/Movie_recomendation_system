# Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on movie metadata including genres, keywords, cast, crew, and plot overview. This project uses TF-IDF vectorization and cosine similarity to find movies with similar content.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Code Architecture & Pipeline Design](#-code-architecture--pipeline-design)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This recommendation system uses **content-based filtering** to recommend movies. Unlike collaborative filtering (which relies on user ratings), this approach analyzes movie content features such as:
- Genres
- Keywords
- Plot overview
- Cast members (top 3 actors)
- Director

The system converts these features into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) and calculates similarity between movies using cosine similarity.

**Note**: The codebase is structured with modular, pipeline-friendly functions that make it easy to extend, maintain, and deploy in production environments. Each component is designed for reusability and can be easily integrated into larger ML pipelines or web services.

## ✨ Features

- **Content-Based Recommendations**: Suggests movies based on content similarity rather than user ratings
- **Comprehensive Feature Extraction**: Combines multiple movie attributes (genres, keywords, cast, crew, overview)
- **TF-IDF Vectorization**: Converts text data into numerical representations
- **Cosine Similarity**: Measures similarity between movies in high-dimensional space
- **Top 10 Recommendations**: Returns the 10 most similar movies for any given movie
- **Modular Pipeline Architecture**: Well-structured, reusable functions designed for easy integration into production systems and ML pipelines

## 📊 Dataset

The project uses the **TMDB 5000 Movie Dataset** from Kaggle, which contains:
- **tmdb_5000_movies.csv**: Movie metadata including genres, keywords, overview, etc.
- **tmdb_5000_credits.csv**: Cast and crew information for each movie

**Dataset Source**: [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 📁 Project Structure

```
Movie Recommendation System/
│
├── Movie_Recommendation_System (2).ipynb  # Main Jupyter notebook
├── README.md                               # Project documentation
├── tmdb_5000_movies.csv                    # Movie metadata (downloaded)
└── tmdb_5000_credits.csv                   # Credits data (downloaded)
```

## 🔧 How It Works

### 1. Data Loading and Preprocessing

1. **Download Dataset**: Downloads the TMDB movie dataset from Kaggle
2. **Merge Datasets**: Combines movies and credits data on the 'title' column
3. **Select Features**: Extracts relevant columns (movie_id, title, genres, keywords, overview, cast, crew)

### 2. Data Processing

The system processes each attribute:

- **Genres**: Extracts all genre names from the genre dictionaries
- **Keywords**: Extracts all keyword names
- **Cast**: Extracts top 3 cast members (most prominent actors)
- **Crew**: Extracts only the Director
- **Overview**: Splits the plot summary into individual words

### 3. Feature Combination

All processed features are combined into a single `tags` column:
```
tags = overview + genres + keywords + cast + crew
```

This creates a comprehensive text representation of each movie.

### 4. Vectorization

- **TF-IDF Vectorization**: Converts the text tags into numerical vectors
  - Each movie is represented as a vector in high-dimensional space
  - TF-IDF weights words based on their importance (frequent in document, rare in corpus)
  - Removes common English stop words

### 5. Similarity Calculation

- **Cosine Similarity**: Calculates similarity between all movie pairs
  - Measures the cosine of the angle between two vectors
  - Values range from 0 (no similarity) to 1 (identical)
  - Creates a symmetric similarity matrix

### 6. Recommendation Generation

For a given movie:
1. Find the movie's index in the dataset
2. Retrieve similarity scores with all other movies
3. Sort movies by similarity (descending)
4. Return top 10 most similar movies (excluding the input movie itself)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- Kaggle API credentials (for downloading the dataset)

### Required Libraries

```bash
pip install pandas numpy scikit-learn kaggle
```

Or install from the notebook:
```python
!pip install kaggle==1.5.12
```

### Setup

1. **Get Kaggle API Credentials**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Scroll to the "API" section
   - Click "Create New Token" to download `kaggle.json`

2. **Configure Kaggle API** (in the notebook):
   ```python
   import os
   os.environ["KAGGLE_USERNAME"] = "your_username"
   os.environ["KAGGLE_KEY"] = "your_api_key"
   ```

3. **Download Dataset**:
   The notebook will automatically download and extract the dataset.

## 💻 Usage

### Running the Notebook

1. Open the Jupyter notebook: `Movie_Recommendation_System (2).ipynb`
2. Run all cells sequentially
3. The system will:
   - Download and load the dataset
   - Preprocess the data
   - Build the recommendation model
   - Be ready to provide recommendations

### Getting Recommendations

```python
# Get recommendations for a movie
recommendations = get_recommendations("Memento")
print(recommendations)
```

**Example Output**:
```
3983                                 Amnesiac
1467                          The Maze Runner
1764                                Dark City
3536                                     Iris
207                              Total Recall
3207                             The I Inside
515                            50 First Dates
2154    Eternal Sunshine of the Spotless Mind
1810                                Self/less
1034                                 Insomnia
Name: title, dtype: object
```

### Function Reference

#### `process_metadata_list(obj, max_items=None, filter_key=None, filter_value=None)`
Unified function to process metadata lists by converting string representations to Python lists and extracting key information.
- **Parameters**:
  - `obj`: String representation of a list/dict or an actual list
  - `max_items`: Maximum number of items to extract (None for all)
  - `filter_key`: Key to filter on (e.g., 'job' for crew members)
  - `filter_value`: Value to match for filter_key (e.g., 'Director')
- **Returns**: List of extracted names/values
- **Design**: Highly reusable and configurable for different metadata types

#### `load_data()`
Loads and preprocesses movie data.
- Returns: DataFrame with columns: `movie_id`, `title`, `tags`
- **Design**: Encapsulates all data preprocessing steps in a single, reusable function

#### `get_recommendations(title, cosine_sim_matrix=cosine_sim, df_data=df)`
Gets movie recommendations based on content similarity.
- **Parameters**:
  - `title`: Movie title to get recommendations for
  - `cosine_sim_matrix`: Pre-computed similarity matrix (optional)
  - `df_data`: Movie DataFrame (optional)
- **Returns**: Series of 10 recommended movie titles
- **Design**: Flexible function that accepts pre-computed matrices for efficiency

## 🏗️ Code Architecture & Pipeline Design

### Modular Function Structure

This project is designed with **modular, pipeline-friendly architecture** that makes it easy to extend, maintain, and integrate into larger systems. The code structure follows best practices for data science pipelines:

#### Key Design Principles

1. **Separation of Concerns**: Each function has a single, well-defined responsibility
   - `process_metadata_list()`: Handles metadata extraction
   - `load_data()`: Manages data loading and preprocessing
   - `get_recommendations()`: Handles recommendation logic

2. **Reusability**: Functions are designed to be reusable across different contexts
   - `process_metadata_list()` can process genres, keywords, cast, or crew with different parameters
   - `load_data()` can be called independently to get clean, processed data
   - `get_recommendations()` accepts pre-computed matrices for efficiency

3. **Pipeline-Ready**: The structure is ideal for building ML pipelines
   ```python
   # Example pipeline structure
   def build_recommendation_pipeline():
       # Step 1: Load and preprocess data
       df = load_data()
       
       # Step 2: Feature engineering
       tfidf_matrix = tfidf.fit_transform(df['tags'])
       
       # Step 3: Model building
       cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
       
       # Step 4: Recommendation service
       return lambda title: get_recommendations(title, cosine_sim, df)
   ```

4. **Extensibility**: Easy to add new features or modify existing ones
   - Add new metadata processing by extending `process_metadata_list()`
   - Modify preprocessing steps in `load_data()` without affecting other components
   - Swap recommendation algorithms while keeping the same interface

5. **Testability**: Each function can be tested independently
   - Unit test `process_metadata_list()` with various inputs
   - Test `load_data()` with mock datasets
   - Test `get_recommendations()` with known similarity matrices

#### Benefits of This Structure

✅ **Maintainability**: Clear separation makes code easy to understand and modify  
✅ **Scalability**: Functions can be parallelized or distributed across systems  
✅ **Integration**: Easy to integrate into web services, APIs, or batch processing systems  
✅ **Version Control**: Individual functions can be versioned and updated independently  
✅ **Debugging**: Issues can be isolated to specific functions  
✅ **Documentation**: Each function is self-contained with clear docstrings  

#### Pipeline Integration Examples

**Example 1: Batch Processing Pipeline**
```python
def batch_recommendations(movie_titles):
    df = load_data()
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    results = {}
    for title in movie_titles:
        results[title] = get_recommendations(title, cosine_sim, df)
    return results
```

**Example 2: API Service Integration**
```python
# Flask/FastAPI example
@app.route('/recommend/<movie_title>')
def recommend(movie_title):
    # Load pre-computed matrices (cached)
    recommendations = get_recommendations(movie_title, cached_cosine_sim, cached_df)
    return jsonify(recommendations.to_dict())
```

**Example 3: Incremental Updates**
```python
def update_recommendations(new_movies_df):
    # Load existing data
    existing_df = load_data()
    
    # Process new movies using the same pipeline
    new_df = process_new_movies(new_movies_df)  # Uses process_metadata_list()
    
    # Combine and rebuild
    combined_df = pd.concat([existing_df, new_df])
    # Rebuild similarity matrix...
```

This architecture makes the codebase **production-ready** and suitable for deployment in real-world applications where maintainability, scalability, and extensibility are crucial.

## 🔬 Methodology

### Content-Based Filtering

This system uses **content-based filtering**, which:
- Analyzes item features (movie content)
- Doesn't require user interaction history
- Provides recommendations based on item similarity
- Works well for new items (cold start problem)

### TF-IDF Vectorization

**Term Frequency-Inverse Document Frequency (TF-IDF)**:
- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **TF-IDF Score**: `TF × IDF` - Higher for important, distinctive words

### Cosine Similarity

**Cosine Similarity** measures the angle between two vectors:
- Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
- Range: 0 to 1 (0 = orthogonal, 1 = identical direction)
- Advantages: Normalized, works well with sparse vectors

## 🛠️ Technologies Used

- **Python 3**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Scikit-learn**: 
  - `TfidfVectorizer`: Text vectorization
  - `cosine_similarity`: Similarity calculation
- **Kaggle API**: Dataset download
- **Jupyter Notebook / Google Colab**: Development environment

## 📈 Results

The system successfully:
- Processes 5000 movies from the TMDB dataset
- Creates TF-IDF vectors for all movies
- Generates meaningful recommendations based on content similarity
- Returns top 10 similar movies for any given movie

**Example**: For "Memento" (a psychological thriller about memory loss), the system recommends similar movies like "Amnesiac", "Eternal Sunshine of the Spotless Mind", and "Total Recall" - all dealing with memory-related themes.

## 🔮 Future Improvements

1. **Hybrid Approach**: Combine content-based and collaborative filtering
2. **User Preferences**: Incorporate user ratings and preferences
3. **Advanced NLP**: Use word embeddings (Word2Vec, GloVe) or transformers
4. **Genre Weighting**: Give more weight to certain features (e.g., genres vs. keywords)
5. **Diversity**: Ensure recommendations are diverse, not just similar
6. **Performance**: Optimize for larger datasets using approximate nearest neighbors
7. **Web Interface**: Create a user-friendly web application
8. **Real-time Updates**: Add new movies and update recommendations dynamically

## 📝 Notes

- The dataset contains approximately 5000 movies
- Movie titles must match exactly (case-sensitive)
- The system works best with movies that have rich metadata
- Processing time increases with dataset size

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of a Machine Learning project for movie recommendation systems.

## 🙏 Acknowledgments

- TMDB (The Movie Database) for providing the dataset
- Kaggle for hosting the dataset
- Scikit-learn team for excellent ML tools

---

**Happy Movie Watching! 🎬🍿**

