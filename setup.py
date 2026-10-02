"""
Brooks Kahsai
CSE 163 AI
Final Project: Data Cleaning and Feature Engineering Pipeline

This script constructs the final analytical dataset used throughout
the study. It combines multiple Spotify data sources, harmonizes
variable definitions, aggregates duplicate observations, removes
missing values, filters infrequent genres and artists, and creates
a genre-standardized popularity metric to facilitate fair
comparisons across musical genres.
"""
import pandas as pd

# -----------------------
# Load + standardize
# -----------------------
files = ["dataset.csv", "dataset2.csv", "dataset3.csv"]

dfs = []
for i, f in enumerate(files, start=1):
    df = pd.read_csv(f)

    df = df.rename(columns={
        "song": "track_name",
        "artists": "artist",
        "track_genre": "genre"
    })

    df.columns = df.columns.str.lower()

    # clean keys
    df["track_name"] = df["track_name"].str.lower().str.strip()
    df["artist"] = df["artist"].str.lower().str.strip()

    # suffix columns except keys
    df = df.rename(columns={
        c: f"{c}_df{i}"
        for c in df.columns
        if c not in ["track_name", "artist"]
    })

    dfs.append(df)

# -----------------------
# Merge
# -----------------------
df = dfs[0].merge(dfs[1], on=["track_name", "artist"], how="outer") \
           .merge(dfs[2], on=["track_name", "artist"], how="outer")

df = df.loc[:, ~df.columns.str.contains("^unnamed")]
df = df.drop_duplicates(subset=["artist", "track_name"])

# -----------------------
# Feature definitions
# -----------------------
numeric_features = [
    "popularity", "duration_ms", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]

non_scale_features = ["duration_ms", "tempo", "time_signature", "key"]
scale_features = list(
    set(numeric_features)
    - set(non_scale_features)
    - {"popularity"}
)

categorical_features = ["genre", "explicit", "album_name", 'artist']

# -----------------------
# Numeric aggregation (vectorized-ish)
# -----------------------
for feat in numeric_features:
    cols = [c for c in df.columns if c.startswith(f"{feat}_df")]

    if cols:
        df[feat] = df[cols].mean(axis=1, skipna=True)
        df.drop(columns=cols, inplace=True)

# -----------------------
# Categorical aggregation
# -----------------------
for feat in categorical_features:
    cols = [c for c in df.columns if c.startswith(f"{feat}_df")]

    if cols:
        df[feat] = df[cols].mode(axis=1, dropna=True)[0]
        df.drop(columns=cols, inplace=True)

df["genre"] = df["genre"].astype(str).str.split(",")

# ----------------------------------
# Remove missing observations
# ----------------------------------

df = df.dropna(
    subset=numeric_features + ["genre"]
)

# ----------------------------------
# Keep top 5 genres
# ----------------------------------

top_genres = (
    df["genre"]
    .explode()
    .value_counts()
    .head(20)
    .index
)


def primary_genre(genres):
    """
    Assign the first genre belonging
    to the top genres. Rewrite the genre data
    entry with desired genre or dummy value "other"
    """
    for genre in genres:
        if genre in top_genres:
            return genre
    return "other"


df["genre"] = df["genre"].apply(primary_genre)
# Keep only top genres
df = df[df["genre"].isin(top_genres)]

# ----------------------------------
# Artist cleanup
# ----------------------------------

df = df.dropna(subset=["artist"])

df["artist"] = (
    df["artist"]
    .str.lower()
    .str.split(",")
)

top_artists = (
    df["artist"]
    .explode()
    .value_counts()
    .head(500)
    .index
)

df = df[df["artist"].apply(
    lambda artists: any(
        artist in top_artists
        for artist in artists
    )
)]

# ----------------------------------
# Genre-standardized popularity
# ----------------------------------

genre_popularity = (df.groupby("genre")["popularity"])


df["popularity_z"] = (
        (df["popularity"] - genre_popularity.transform("mean")
         ) / genre_popularity.transform("std"))

# ----------------------------------
# Final cleanup
# ----------------------------------

df = df.drop(
    columns=[
        c for c in df.columns
        if c.endswith((
            "_df1",
            "_df2",
            "_df3"
        ))
    ],
    errors="ignore"
)
