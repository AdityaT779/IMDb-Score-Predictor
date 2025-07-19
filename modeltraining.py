import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import joblib
from feature_engine.encoding import CountFrequencyEncoder

# Load data
df = pd.read_csv("/Users/atulkumartiwary/Downloads/ultimateset2.csv")

# feature engineering
df['genre_count'] = df['genres'].str.split('|').str.len()
df['genre_count'] = np.where(df['genre_count'] > 5, 5, df['genre_count'])

df['b/m'] = df['budget'] / (df['duration'].replace(0, 1) + 0.1)
df['log_budget'] = np.log1p(df['budget'])
df['log_bm'] = np.log1p(df['b/m'])  # Avoid negative values

director_avg = df.groupby('director_name')['imdb_score'].mean()
df['director_avg_score'] = df['director_name'].map(director_avg)

director_count = df.groupby('director_name')['imdb_score'].transform('count')  # Number of movies per director
global_mean = df['imdb_score'].mean()
df['director_avg_score'] = (
    (df['director_avg_score'] * director_count + global_mean * 5) / (director_count + 5)
)

df['director_avg_score'] = df['director_avg_score'].fillna(global_mean)

df['director_movie_count'] = director_count  # Could also log-transform this

df['collab'] = df.groupby(['director_name', 'actor_1_name'])['imdb_score'].transform('count')

df['year_squared'] = df['title_year'] ** 2

director_stats_path = "/Users/atulkumartiwary/Desktop/pythonstuff/modeltrial/moviepredic(existing dataset)/director_stats.csv"
director_stats = df[['director_name', 'director_avg_score', 'director_movie_count']].drop_duplicates()
director_stats.to_csv(director_stats_path, index=False)
print(f"Saved director stats to {director_stats_path}")

features = [
    'log_budget', 'duration', 'language', 'genres', 'title_year',
    'content_rating', 'director_avg_score','director_movie_count', 'actor_1_name', 'log_bm',
    'year_squared', 'genre_count','collab'
]
target = 'imdb_score'

df = df[features + [target]].dropna(subset=[target])
X = df[features]
y = df[target]

#selecting features for preprocessing
num_cols = ['director_avg_score', 'director_movie_count','log_budget', 'duration', 'title_year', 'log_bm', 'year_squared', 'genre_count','collab']
cat_cols = ['language', 'content_rating', 'genres']
high_card_cols = ['actor_1_name']

# Preprocessing using pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

freq_encode_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', CountFrequencyEncoder(encoding_method='frequency'))
])

onehot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('freq_enc', freq_encode_pipeline, high_card_cols),
    ('onehot', onehot_pipeline, cat_cols)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42))
])

# Hyperparameters 
param_dist = {
    'regressor__num_leaves': [15, 31, 63],  # Wider range
    'regressor__min_data_in_leaf': [20, 50, 100],  # Alternative to min_child_samples
    'regressor__learning_rate': [0.005, 0.01, 0.05],  # Include smaller values
    'regressor__feature_fraction': [0.6, 0.8, 0.9],
    'regressor__bagging_fraction': [0.8, 0.9],  # New
    'regressor__lambda_l1': [0, 0.1, 0.5],  # Wider range
    'regressor__lambda_l2': [0, 0.1]  # Add L2 regularization
}

#model splitting and trainining
train_cutoff = 2022  # Change this year as needed

train_idx = df['title_year'] <= train_cutoff
valid_idx = df['title_year'] > train_cutoff

X_train, y_train = X[train_idx], y[train_idx]
X_valid, y_valid = X[valid_idx], y[valid_idx]


random_search = RandomizedSearchCV(
    model_pipeline,
    n_iter=30,
    param_distributions=param_dist,
    scoring='neg_mean_absolute_error',
    cv=3,  
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Starting training with time-based split (train: <=2022, valid: 2023-2025)...")
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)

final_model = random_search.best_estimator_
joblib.dump(final_model, "/Users/atulkumartiwary/Desktop/pythonstuff/modeltrial/moviepredic(existing dataset)/imdb_score_predictor_future_pipeline.pkl")

feature_names = final_model.named_steps['preprocessor'].get_feature_names_out()
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': final_model.named_steps['regressor'].feature_importances_
}).sort_values(by='Importance', ascending=False)

# EVALUATE ON HELD-OUT "FUTURE" DATA (2023-2025)

y_pred = final_model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("\nTraining complete! Results:")
print("Top 10 Features:")
print(importances.head(10))
print(f"\nValidation MAE (2023-2025): {mae:.3f}")
print(f"Validation R² (2023-2025): {r2:.3f}")
print(f"Validation RMSE (2023-2025): {rmse:.3f}")

