# IMDb-Score-Predictor

Predict the IMDB score of future movies using machine learning, modern feature engineering, and a streamlined web interface. Enter your own movie details and get an instant, data-driven prediction.!

# What is this project?

This is an interactive app and pipeline for predicting the IMDB rating of unreleased or hypothetical movies. Built on a LightGBM regression model, it combines data preprocessing, feature engineering, and hyperparameter tuning for robust predictions. Just fill in a few movie details (genre, budget, director, star, language, release year, etc.) and the model will provide a likely IMDB score.

## ✨ Features

- User-friendly app: Enter movie details in a Streamlit interface—no code required!
- Achieves a mean absolute error (MAE) of 0.47 and an R² score of 0.49 on held-out "future" data (2023-2025).
- Smart feature engineering: Incorporates director and actor histories, genre breakdown, budget scaling, and more for each prediction.

## 🛠️ How does it work?

1. The model is trained on 5,000 movies, each with extensive details: budget, genres, runtime, language, release year, rating, director, main actor, and more.
2. Computes new features like average director score, collaboration counts, budget-per-minute, genre diversity, and nonlinear year effects.
3. Uses LightGBM regression, hyperparameter optimization (RandomizedSearchCV), and advanced encoding for high-cardinality features.
4. Evaluated with a time-based train/validation split for realistic performance.
5. Enter details via the web UI; the backend applies the same processing pipeline and returns a score between 1 and 10.

## 🏆 Current Performance

| Metric        | Score       |
|---------------|-------------|
| MAE           | 0.472       |
| R²            | 0.49        |

*Tested on future years (2023-2025) for real-world generalization.*

## 🤖 Try it!

1. Clone the repo.
2. Install requirements.
3. Run the Streamlit app (`streamlit run app.py`).
4. Enter your movie details—get a predicted IMDB score instantly!

## 🚧 Ongoing Improvements

I'm continuously working to make the model smarter by:
- Adding more data to further improve accuracy.
- Testing new features (cast chemistry, genre interactions, etc.). Currently relies a lot on director status, trying to move away from that
- Experimenting with different model architectures and validation schemes.
