import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
movies = pd.read_csv('movies.csv')  # Use the movies.csv from MovieLens dataset

# Handle missing genres
movies['genres'] = movies['genres'].fillna('')

# Convert genres like "Action|Adventure|Comedy" to "Action Adventure Comedy"
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# TF-IDF vectorizer to convert genres to numerical format
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a reverse mapping of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found in dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 excluding the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example usage
if __name__ == "__main__":
    movie_name = input("Enter a movie title: ")
    print("\nTop 5 recommendations based on genres:\n")
    recommendations = recommend(movie_name)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for i, title in enumerate(recommendations, start=1):
            print(f"{i}. {title}")
