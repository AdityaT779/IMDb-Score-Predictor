import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
pipeline = joblib.load("/Users/atulkumartiwary/Desktop/pythonstuff/modelfinals/imdbpredictor/imdb_score_predictor_future_pipeline.pkl")

# Loading  director stats from training data

try:
    director_stats = pd.read_csv("director_stats.csv")
    global_mean = director_stats['director_avg_score'].mean()
except:
    director_stats = pd.DataFrame(columns=['director_name', 'director_avg_score', 'director_movie_count'])
    global_mean = 6.5  #  average IMDB score

st.title("IMDB Score Predictor")
st.write("Enter your movie details:")

# User inputs
budget = st.number_input("Budget (USD):", min_value=0, step=1000000)
duration = st.number_input("Duration (minutes):", min_value=1, step=1, value=120)
genres = st.text_input("Genres (separated by |, e.g., Action|Comedy):", value="Action|Comedy")
language = st.text_input("Language:", value="English")
title_year = st.number_input("Release year:", min_value=1900, step=1, value=2023)
content_rating = st.text_input("Age Rating (e.g., PG-13, R):", value="PG-13")
director_name = st.text_input("Director:", value="Christopher Nolan")
actor_1_name = st.text_input("Actor 1 Name:", value="Leonardo DiCaprio")

# Feature engineering (matches training code)
genre_count = min(len(genres.split('|')), 5)  
log_budget = np.log1p(budget)
b_m = budget / (duration if duration > 0 else 1 + 0.1)
log_bm = np.log1p(b_m)
year_squared = title_year ** 2

# Handle director features
director_lower = director_name.lower().strip()
director_match = director_stats[director_stats['director_name'].str.lower() == director_lower]

if not director_match.empty:
    # Director exists in  database
    director_avg = director_match['director_avg_score'].values[0]
    director_count = director_match['director_movie_count'].values[0]
    collab_count = 0  
else:
    # New director - use global mean and minimum count
    director_avg = global_mean
    director_count = 1  # Minimum count for new directors
    collab_count = 0

#final input data to be sed
input_data = pd.DataFrame({
    'log_budget': [log_budget],
    'duration': [duration],
    'language': [language.lower().strip()],
    'genres': ['|'.join(sorted(g.lower().strip() for g in genres.split('|')))],  # Match training format
    'title_year': [title_year],
    'content_rating': [content_rating.lower().strip()],
    'director_avg_score': [director_avg],
    'director_movie_count': [director_count],
    'actor_1_name': [actor_1_name.lower().strip()],
    'log_bm': [log_bm],
    'year_squared': [year_squared],
    'genre_count': [genre_count],
    'collab': [collab_count]
})

if st.button("Predict IMDB Score"):
    try:
        prediction = pipeline.predict(input_data)[0]
        prediction = max(1, min(10, round(prediction, 1)))
        st.success(f"Predicted IMDB Score: {prediction:.1f}")
        
        # Show director info if available
        if not director_match.empty:
            st.info(f"Director stats: {director_count} movies in database, average score {director_avg:.1f}")
        else:
            st.info("Director not in database - using average values")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")