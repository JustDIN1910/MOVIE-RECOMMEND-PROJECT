from flask import Flask, render_template, request
import pandas as pd
from tmdbv3api import TMDb, Movie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = '1cca496bc1e952bf8573629134cf7b51'  # Replace with your TMDb API Key
movie_api = Movie()

# Fetch top movies from TMDb
def fetch_movies():
    try:
        print("Fetching top movies from TMDb...")
        # Fetch popular movies (You can customize the type of movies here)
        popular_movies = movie_api.popular(page=1)
        
        movie_list = []
        for movie in popular_movies:
            try:
                movie_details = movie_api.details(movie.id)
                description = movie_details.get('overview', 'No Description')  # Ensure this field exists
                genres = [genre['name'] for genre in movie_details.get('genres', [])]
                year = movie_details.get('release_date', '')[:4]  # Extract the year from release_date (e.g., '2024-11-30')
                movie_list.append({
                    'title': movie_details.get('title', 'No Title'),
                    'genres': ', '.join(genres) if genres else 'No Genres',
                    'description': description,
                    'year': int(year) if year else 0  # Ensure year is an integer
                })
            except Exception as e:
                print(f"Error fetching details for {movie}: {e}")
        
        df = pd.DataFrame(movie_list)

        if df.empty:
            print("Error: No movie data fetched.")
        else:
            # Debugging: Check columns and first few rows
            print(df.columns)  # Print the column names
            print(df.head())    # Print the first few rows
        return df
    except Exception as e:
        print(f"Error fetching top movies: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Build the recommendation system
def build_recommendation_system(df):
    if df.empty:
        print("Error: No data to build the recommendation system.")
        return None, None
    
    # Handle missing descriptions
    df['description'] = df.get('description', pd.Series([''] * len(df))).fillna('')

    # Add content for recommendation based on genres and descriptions
    df['content'] = df['genres'] + " " + df['description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

# Function to recommend movies based on genre and year
def recommend_movies_by_genre(genre, year_range, cosine_sim, indices, df):
    # Split the year range into start and end years
    start_year, end_year = map(int, year_range.split('-'))

    # Filter movies by genre
    genre_movies = df[df['genres'].str.contains(genre, case=False, na=False)]
    
    # Further filter by year range
    genre_movies = genre_movies[genre_movies['year'].between(start_year, end_year)]

    if genre_movies.empty:
        print(f"No movies found for genre: {genre} and year range: {year_range}")
        return ["No movies found for this genre and year range."]
    
    print(f"Movies found for genre {genre} and year range {year_range}:")
    print(genre_movies[['title', 'genres', 'year']])  # Print the genre, title, and year for debugging
    
    # Get recommendations based on cosine similarity
    recommendations = []
    for title in genre_movies['title']:
        idx = indices.get(title, None)
        if idx is None:
            continue
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Show top 5 recommendations
        movie_indices = [i[0] for i in sim_scores]
        recommendations.extend(df['title'].iloc[movie_indices].tolist())

    return recommendations[:10]  # Limit the number of recommendations to 10

# Load movie data
movie_data = fetch_movies()
print(f"Movie Data Loaded: {not movie_data.empty}")  # Check if data was fetched successfully

cosine_sim, indices = None, None
if not movie_data.empty:
    cosine_sim, indices = build_recommendation_system(movie_data)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    genre = request.form['genre']
    year_range = request.form['year_range']  # Get the year range selected by the user
    recommendations = []
    
    # Debugging: Check if recommendation system is initialized properly
    if cosine_sim is None or indices is None:
        print("Recommendation system not initialized!")
        recommendations = ["Error: Recommendation system is not available."]
    else:
        recommendations = recommend_movies_by_genre(genre, year_range, cosine_sim, indices, movie_data)
    
    return render_template('results.html', genre=genre, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
