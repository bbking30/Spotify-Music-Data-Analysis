"""
Brooks Kahsai
CSE 163 AI
Final Project: Q4 Code

Comparison of Audio Features by Explicit Content

This script examines how Spotify audio features differ between
explicit and non-explicit tracks. For each numeric feature, a
separate ordinary least squares regression is fitted to estimate
the mean difference associated with explicit content.

Results are visualized using bar plots showing average feature
values by explicit category, with regression-derived effect sizes
and p-values displayed on each subplot for quick interpretation.
"""
from setup import df, numeric_features
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import test

fig, axes = plt.subplots(3, 5, figsize=(20, 18))
axes = axes.flatten()

# Copy dataframe
df_copy = df.copy()
df_copy["explicit"].replace({True: "Explicit", False: "Non-Explicit"})
test.copy_test(df_copy)

for ax, feature in zip(axes.flatten(), numeric_features):
    # Run OLS Regression Model to get Stats on Fit
    model = smf.ols(f"{feature} ~ explicit", data=df).fit()
    coef = model.params["explicit[T.True]"]
    p_val = model.pvalues["explicit[T.True]"]

    # Plot
    palette = {False: "steelblue", True: "tomato"}

    sns.barplot(
        data=df_copy,
        x="explicit",
        y=feature,
        hue="explicit",
        palette=palette,
        legend=False,
        ax=ax
    )

    fig.suptitle("Feature Means by Explicit Content",
                 fontsize=16)
    ax.set_xlabel("")

    # Add OLS Regression Model Results to Plots
    ax.text(
        0.5, 0.95,
        f"Δ={coef:.3f}\np={p_val:.2g}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9
    )

# Display Results
plt.tight_layout()
plt.show()
