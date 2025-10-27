import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict, Any

def load_data(filename: str = "netflix_titles.csv") -> Optional[pd.DataFrame]:
    """Loads the Netflix titles dataset."""
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure 'netflix_titles.csv' is in the correct directory.")
        return None

class ContentRecommender:
    """
    A content-based recommendation system using TF-IDF and Cosine Similarity.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cosine_sim: Optional[Any] = None
        self.indices: Optional[pd.Series] = None
        self.model_setup()

    def _preprocess_features(self) -> None:
        """Fills NaNs and combines description and genre features."""
        # Fill NaNs in 'description' and 'listed_in' (genres)
        self.df['description'] = self.df['description'].fillna('')
        self.df['listed_in'] = self.df['listed_in'].fillna('')
        
        # Combine the relevant features for vectorization
        self.df['combined_features'] = self.df['description'] + ' ' + self.df['listed_in']

    def model_setup(self) -> None:
        """Applies TF-IDF vectorization and computes the Cosine Similarity matrix."""
        print("Setting up recommender model (TF-IDF vectorization and Cosine Similarity)...")
        self._preprocess_features()
        
        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the data
        tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        
        # Compute Cosine Similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping of title to index
        # Resetting index ensures a clean, 0-based integer index for similarity matrix lookup
        self.df = self.df.reset_index(drop=True)
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        print("Setup complete.")

    def get_content_recommendations(self, seed_title: str, N: int = 10) -> List[str]:
        """
        Generates N content-based recommendations for a given seed title.
        
        Args:
            seed_title: The title of the content to find similar items for.
            N: The number of recommendations to return.
            
        Returns:
            A list of recommended titles.
        """
        if self.indices is None or self.cosine_sim is None:
            return ["Error: Recommender model not properly set up."]

        try:
            # Get the index corresponding to the seed_title
            idx = self.indices[seed_title]
        except KeyError:
            return [f"Error: Title '{seed_title}' not found in the dataset."]

        # Get the pairwise similarity scores for all items with that content
        # Convert the similarity array to a Series for easier sorting and manipulation
        sim_scores = pd.Series(self.cosine_sim[idx])
        
        # Sort the items and get the top N+1 (including the seed item itself)
        # Use nlargest to get the top N scores, excluding the score at index 'idx' (the item itself)
        recommended_indices = sim_scores.drop(idx).nlargest(N).index
        
        # Return the titles corresponding to the top indices
        return self.df['title'].iloc[recommended_indices].tolist()

def main():
    """Main function to run the recommendation system."""
    df = load_data()

    if df is not None:
        recommender = ContentRecommender(df)

        # SEED
        SEED_TITLE = "The Queen's Gambit"
        
        print("\n" + "="*50)
        print(f"Content-Based: Top {10} Similar to '{SEED_TITLE}'")
        print("="*50)

        recommendations = recommender.get_content_recommendations(
            SEED_TITLE, N=10
        )

        if isinstance(recommendations, list) and not recommendations[0].startswith("Error"):
            for i, title in enumerate(recommendations, 1):
                print(f"{i}. {title}")
        else:
            # Print the error message if one was returned
            print(recommendations[0]) 

if __name__ == "__main__":
    main()