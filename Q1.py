"""
Brooks Kahsai
CSE 163 AI
Final Project: Q1 Code

Exploratory Regression Analysis of Audio Features vs Popularity

This script investigates the relationship between Spotify audio features
and standardized track popularity using simple linear regression models.

The dataset is aggregated at the artist level to reduce overplotting and
mitigate bias from artists with many tracks. For each predictor, an OLS
model is fitted against popularity z-scores, and results are visualized
using scatterplots with regression lines. Model coefficients, R² values,
and p-values are displayed for quick comparison across features.
"""
from setup import df
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import test


PREDICTORS = [
    "duration_ms",
    "energy",
    "tempo",
    "valence",
    "speechiness",
    "danceability",
    "acousticness",
    "loudness",
]

TARGET = "popularity_z"

# Aggregate by artist to reduce overplotting and prevent artists with
# many tracks from dominating the analysis.
df_copy = df.copy()
test.copy_test(df_copy)

df_copy = df_copy.explode("artist")
df_copy = df_copy.groupby(
    "artist",
    as_index=False
)[PREDICTORS + [TARGET]].mean()

test.subset_test(df_copy, PREDICTORS + [TARGET] + ['artist'])

# Create a grid of plots showing each audio feature against popularity.
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, feature in enumerate(PREDICTORS):
    row, col = divmod(i, 4)

    # Fit a simple linear regression model.
    model = smf.ols(
        f"{TARGET} ~ {feature}",
        data=df_copy
    ).fit()

    coef = model.params[feature]
    p_val = model.pvalues[feature]
    r_squared = model.rsquared

    # Plot the relationship and fitted regression line.
    sns.regplot(
        data=df_copy,
        x=feature,
        y=TARGET,
        ax=axes[row, col],
        scatter_kws={"alpha": 0.3},
    )

    axes[row, col].set_title(feature)

    # Use scientific notation for track duration only.
    if feature == "duration_ms":
        axes[row, col].ticklabel_format(
            style="sci",
            axis="x",
            scilimits=(0, 0),
        )

    # Only label the y-axis on the leftmost plots.
    if col == 0:
        axes[row, col].set_ylabel("Popularity (z-score)")
    else:
        axes[row, col].set_ylabel("")

    axes[row, col].set_xlabel("")

    # Display model statistics for quick interpretation.
    axes[row, col].text(
        0.5,
        0.95,
        (
            f"β={coef:.3f}\n"
            f"R²={r_squared:.3f}\n"
            f"p={p_val:.2g}"
        ),
        transform=axes[row, col].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

# Add title for full figure
fig.suptitle(
    "Audio Features vs. Standardized Popularity",
    fontsize=16,
)

# Display Result
plt.tight_layout()
plt.show()
