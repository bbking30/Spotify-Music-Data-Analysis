"""
Brooks Kahsai
CSE 163 AI
Final Project: Q2 Code

Predictive Modeling of Spotify Track Popularity

This script builds and evaluates multiple machine learning models
to predict Spotify track popularity using audio features and
metadata.

Two feature settings are compared:
1) Audio-only features (continuous audio descriptors)
2) Full feature set including audio features plus artist and genre
   identity encoded via one-hot encoding.

The dataset is split into training and testing sets, and features
are standardized where appropriate. Three models are evaluated:
Lasso regression with cross-validation, Random Forest regression,
and a Multi-Layer Perceptron neural network.

Model performance is assessed using R² and Mean Squared Error,
and Lasso sparsity is reported via the number of nonzero coefficients.
"""
from setup import df
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import test

# -----------------------
# CONFIG
# -----------------------
TARGET = "popularity"

audio_features = [
    "duration_ms", "energy", "tempo", "valence",
    "speechiness", "danceability", "acousticness",
    "loudness"
]

df_copy = df.copy()
test.copy_test(df_copy)

# -----------------------
# OPTIONAL: CLEAN MULTI-LABELS
# -----------------------
df_copy = df_copy.explode("artist").explode("genre")

artist_dummies = pd.get_dummies(df_copy["artist"], prefix="artist")
genre_dummies = pd.get_dummies(df_copy["genre"], prefix="genre")

# -----------------------
# BUILD FEATURE SETS
# -----------------------

df_copy = df_copy.drop(
    columns=[
        "artist",
        "genre",
        "album_name",
        "track_name",
        'popularity_z'
    ], errors="ignore"
)

df_audio = df_copy[audio_features + [TARGET]].dropna()

df_full = (pd.concat(
    [df_copy, artist_dummies, genre_dummies], axis=1)
           .dropna(subset=[TARGET]))

# Run tests to ensure we have properly filtered subsets of the data
test.subset_test(df_audio, audio_features + [TARGET])

# Don't assert dataframe width because we've already expanded
# artist names, but check for other consistencies
test.subset_test(df_full, audio_features + [TARGET], check=False)


# -----------------------
# TRAIN / TEST SPLIT
# -----------------------


def split(df):
    """
    Split a dataset into training and testing sets for supervised learning.
    This function separates the target variable from numeric predictor
    features, fills missing values with zeros, and returns an 80/20
    train-test split with a fixed random seed for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing both features and the target variable.
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test resulting from train_test_split.
    """
    X = df.drop(columns=[TARGET]).select_dtypes(include="number").fillna(0)
    y = df[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42)


X_train_a, X_test_a, y_train_a, y_test_a = split(df_audio)
X_train_f, X_test_f, y_train_f, y_test_f = split(df_full)

# -----------------------
# SCALING (ONLY WHERE NEEDED)
# -----------------------
scaler = StandardScaler()

X_train_a_scaled = scaler.fit_transform(X_train_a)
X_test_a_scaled = scaler.transform(X_test_a)

X_train_f_scaled = scaler.fit_transform(X_train_f)
X_test_f_scaled = scaler.transform(X_test_f)

# -----------------------
# EVALUATION HELPER
# -----------------------


def evaluate(name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Evaluate and print regression model performance metrics.
    This function computes and displays common regression evaluation
    metrics, including R² (coefficient of determination) and Mean Squared
    Error (MSE), for a given set of true and predicted values.
    """
    print(f"\n=== {name} ===")
    print("R²:", round(r2_score(y_true, y_pred), 4))
    print("MSE:", round(mean_squared_error(y_true, y_pred), 4))


# -----------------------
# MODELS
# -----------------------


def run_lasso(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Train and evaluate a Lasso regression model with cross-validation.

    This function fits a LassoCV model using cross-validation to select
    the optimal regularization strength. It then generates predictions on
    the test set and evaluates performance using R² and MSE. The number
    of non-zero coefficients is also reported to indicate model sparsity.
    """
    # Run model
    model = LassoCV(
        cv=5,
        alphas=np.logspace(-4, 1, 50),
        max_iter=5000,
        random_state=42
    )

    # Fit model and compute predictions
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Print results
    evaluate("LASSO", y_test, preds)
    print("Nonzero coefficients:", np.sum(model.coef_ != 0))


def run_rf(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Train and evaluate a Random Forest regression model.

    This function fits a Random Forest regressor to the training data
    and evaluates predictive performance on the test set using R² and
    Mean Squared Error (MSE). The model captures nonlinear relationships
    using an ensemble of decision trees.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    evaluate("RANDOM FOREST", y_test, preds)


def run_mlp(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Train and evaluate a Multi-Layer Perceptron regression model.

    This function fits a feedforward neural network (MLPRegressor)
    with two hidden layers to model nonlinear relationships between
    features and target values. Performance is evaluated on the test
    set using R² and Mean Squared Error (MSE).
    """
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    evaluate("MLP", y_test, preds)

# -----------------------
# RUN: AUDIO ONLY
# -----------------------
print("\n\n===== AUDIO FEATURES ONLY =====")
run_lasso(X_train_a_scaled, X_test_a_scaled, y_train_a, y_test_a)
run_rf(X_train_a, X_test_a, y_train_a, y_test_a)
run_mlp(X_train_a_scaled, X_test_a_scaled, y_train_a, y_test_a)

# -----------------------
# RUN: FULL MODEL (AUDIO + IDENTITY)
# -----------------------
print("\n\n===== AUDIO + ARTIST + GENRE =====")
run_lasso(X_train_f_scaled, X_test_f_scaled, y_train_f, y_test_f)
run_rf(X_train_f, X_test_f, y_train_f, y_test_f)
run_mlp(X_train_f_scaled, X_test_f_scaled, y_train_f, y_test_f)
