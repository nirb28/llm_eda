import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a connection to the SQLite database
conn = sqlite3.connect('sample_movies.db')

# Create movies data
np.random.seed(42)
n_movies = 100

# Generate movie data
movies_data = {
    'movie_id': range(1, n_movies + 1),
    'title': [f"Movie {i}" for i in range(1, n_movies + 1)],
    'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance'], n_movies),
    'release_year': np.random.randint(1990, 2024, n_movies),
    'budget': np.random.randint(1000000, 200000000, n_movies),
    'revenue': np.random.randint(500000, 500000000, n_movies)
}

# Create ratings data
n_ratings = 1000
ratings_data = {
    'rating_id': range(1, n_ratings + 1),
    'movie_id': np.random.randint(1, n_movies + 1, n_ratings),
    'user_id': np.random.randint(1, 1000, n_ratings),
    'rating': np.random.uniform(1, 5, n_ratings).round(1),
    'rating_date': [(datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d') 
                   for _ in range(n_ratings)]
}

# Create DataFrames
movies_df = pd.DataFrame(movies_data)
ratings_df = pd.DataFrame(ratings_data)

# Save to SQLite database
movies_df.to_sql('movies', conn, index=False, if_exists='replace')
ratings_df.to_sql('ratings', conn, index=False, if_exists='replace')

# Create some views for easier analysis
conn.execute("""
CREATE VIEW IF NOT EXISTS movie_stats AS
SELECT 
    m.movie_id,
    m.title,
    m.genre,
    m.release_year,
    m.budget,
    m.revenue,
    COUNT(r.rating_id) as num_ratings,
    ROUND(AVG(r.rating), 2) as avg_rating,
    (m.revenue - m.budget) as profit
FROM movies m
LEFT JOIN ratings r ON m.movie_id = r.movie_id
GROUP BY m.movie_id
""")

conn.commit()
conn.close()

print("Sample database created successfully!")
