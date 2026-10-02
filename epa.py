"""
Brooks Kahsai
CSE 163 AI
Final Project: Exploratory Data Analysis

Computes and displays seven-number summaries for selected numerical
variables in the Spotify dataset. Reported statistics include the
minimum, first quartile, median, third quartile, maximum, mean,
standard deviation, and observation count, enabling preliminary
assessment of variable distributions and potential outliers.
"""
import pandas as pd
from setup import df


def seven_number_summary(df: pd.DataFrame, column: str) -> None:
    """
    Compute and print descriptive statistics for a numeric variable.
    The function removes missing values and reports the observation
    count, mean, standard deviation, minimum, first quartile,
    median, third quartile, and maximum of the specified column.
    """
    col = df[column].dropna()
    summary = {
        "mean": col.mean(),
        "count": col.count(),
        "std": col.std(),
        "min": col.min(),
        "q1": col.quantile(0.25),
        "median": col.median(),
        "q3": col.quantile(0.75),
        "max": col.max()
    }
    print(column, "\n")
    for key in summary.keys():
        print(f'{key}: {summary[key]}')
    print()


for col in [
    'duration_ms',
    'energy',
    'tempo',
    'valence',
    'speechiness',
    'danceability',
    'acousticness',
    'loudness',
    'popularity',
    'popularity_z'
]:
    seven_number_summary(df, col)
