import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(filename="netflix_titles.csv"):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        print("Please make sure 'netflix_titles.csv' is in the same directory.")
        return None

def get_content_recommendations(seed_title, df, cosine_sim, indices, N=10):
    try:
        idx = indices[seed_title]
    except KeyError:
        return f"Error: Title '{seed_title}' not found in the dataset."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]

    item_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[item_indices].tolist()

def main():
    df = load_data()

    if df is not None:
        # 1. Combine relevant features into a single string
        df['combined_features'] = df['description'].fillna('') + ' ' + df['listed_in'].fillna('')

        # 2. Apply TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])

        # 3. Compute Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        df = df.reset_index()
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()

        # SEED
        SEED_TITLE = "The Queen's Gambit"

        print(f"Content-Based: Top 10 Similar to '{SEED_TITLE}'")
        recommendations = get_content_recommendations(
            SEED_TITLE, df, cosine_sim, indices, N=10
        )

        if isinstance(recommendations, list):
            for i, title in enumerate(recommendations, 1):
                print(f"{i}. {title}")
        else:
            print(recommendations)

if __name__ == "__main__":
    main()