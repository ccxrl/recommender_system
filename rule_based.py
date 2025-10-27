import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['type', 'listed_in', 'date_added', 'rating', 'title'], inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df = df.dropna(subset=['date_added'])
    return df


def rule_1_recent_movies(df, top_n=10):
    # Rule 1:
    filtered = df[
        (df['type'] == 'Movie') &
        (df['rating'].isin(['PG-13', 'TV-MA']))
    ].sort_values(by='date_added', ascending=False)

    return filtered[['title', 'rating', 'release_year', 'date_added']].head(top_n)


def rule_2_international_tvshows(df, top_n=10):
    # Rule 2:
    filtered = df[
        (df['type'] == 'TV Show') &
        (df['listed_in'].str.contains('International TV Shows', case=False, na=False))
    ].sort_values(by='date_added', ascending=False)

    return filtered[['title', 'listed_in', 'release_year', 'date_added', 'rating']].head(top_n)

def rule_3_top_10_recent(df, top_n=10):
    # Rule 3: Top 10 most recently added titles (movies or TV shows)
    filtered = df.sort_values(by='date_added', ascending=False)
    return filtered[['title', 'type', 'release_year', 'date_added']].head(top_n)


def print_recommendations(df, title):
    print(f"\n{title}")
    for idx, (i, row) in enumerate(df.iterrows(), start=1):
        print(f"{idx}. {row['title']} ({row['release_year']})")
        if 'type' in df.columns:
            print(f"   Type: {row['type']}")
        if 'rating' in df.columns:
            print(f"   Rating: {row['rating']}")
        if 'listed_in' in df.columns:
            print(f"   Category: {row['listed_in']}")
        print(f"   Date Added: {row['date_added'].date()}")
        print()


def main():
    filepath = 'netflix_titles.csv'
    df = load_data(filepath)

    # Rule 1 Output
    movies = rule_1_recent_movies(df)
    print_recommendations(movies, "Top 10 Recently Added Movies (PG-13 or TV-MA)")

    # Rule 2 Output
    shows = rule_2_international_tvshows(df)
    print_recommendations(shows, "Top 10 International TV Shows")

    # Rule 3 Output
    top_10 = rule_3_top_10_recent(df)
    print_recommendations(top_10, "Top 10 Most Recently Added Titles (Overall)")


if __name__ == "__main__":
    main()
