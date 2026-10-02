"""
Brooks Kahsai
CSE 163 AI
Final Project: Q3 Code

Genre-Level Comparison of Standardized Audio Features

This script generates a heatmap comparing average audio feature
values across music genres in the Spotify dataset.

After cleaning missing values, all numeric features are scaled
using Min-Max normalization to ensure comparability across
different measurement ranges. The data is then aggregated by
genre, computing mean feature values for each group.

A heatmap is used to visualize cross-genre differences in
audio characteristics, with values constrained to [0, 1] to
reflect standardized feature scales.
"""
from setup import df, numeric_features, scale_features
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import test

# 1. ensure numeric only
df_copy = df.copy()
test.copy_test(df_copy)
df_copy = df_copy.dropna(subset=numeric_features + ["genre"])
# 2. standardize
scaled = df_copy.copy()
scaled[numeric_features] = MinMaxScaler().fit_transform(
    scaled[numeric_features])

# Run sanity test to ensure values range from 0-1
test.range_sanity_test(scaled[numeric_features])

# 3. proper groupby mean table
heatmap_data = scaled.groupby("genre")[scale_features].mean()

# 4. force numeric matrix (prevents sneaky object dtype issues)
heatmap_data = heatmap_data.astype(float)

# Make sure we have a properly scaled subset of the data
test.subset_test(heatmap_data, scale_features)

# 5. plot
fig, ax = plt.subplots()
sns.heatmap(
    data=heatmap_data,
    cmap="rocket",
    ax=ax,
    vmin=0,
    vmax=1,
    linewidths=2,
    linecolor="white"
)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_ylabel("Genre")
ax.set_xlabel("Audio Features", labelpad=15)

plt.title("Average Standardized Features by Genre")
plt.tight_layout()
plt.show()
