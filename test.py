"""
Brooks Kahsai
CSE 163 AI
Final Project: Dataset Validation Test Suite

This module contains a set of assertion-based tests used to validate
the integrity and consistency of the Spotify dataset throughout the
data analysis pipeline.

The tests ensure that:
- The dataset retains its expected baseline structure (shape and columns)
- No missing values are introduced during preprocessing
- Subsets of features contain expected columns and dimensions
- Genre filtering remains consistent with predefined valid genres
- MinMax-scaled features remain within the valid [0, 1] range (with
  tolerance for floating-point precision)

These checks are designed to be run at key stages of the analysis
workflow to prevent silent data corruption and ensure reproducibility
of results across different research questions.
"""
from setup import df, top_genres
import pandas as pd

BASELINE_SHAPE = df.shape
BASELINE_COLUMNS = set(df.columns)


def missing_values_test(df: pd.DataFrame) -> None:
    """
    Assert that the inputted dataset contains no missing values.

    Checks the entire DataFrame for NaN values and raises an
    AssertionError if any are present.
    """
    assert df.isna().sum().sum() == 0, \
        "Dataset still contains missing values."


def copy_test(df: pd.DataFrame) -> None:
    """
    Assert that the inputted dataset is an exact copy of the
    dataframe from setup.

    Checks entire dataframe for inconsistencies in shape
    and column designations
    """
    assert df.shape == BASELINE_SHAPE, (
        f"Shape mismatch: expected {BASELINE_SHAPE}, got {df.shape}"
    )
    missing = set(BASELINE_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(BASELINE_COLUMNS)
    assert not missing, f"Missing columns: {missing}"
    assert not extra, f"Unexpected columns: {extra}"
    missing_values_test(df)
    genre_test(df)


def subset_test(
        df: pd.DataFrame,
        columns: list[str],
        check: bool = True
) -> None:
    """
    Validate that a DataFrame contains an expected subset of columns.
    Ensures that all specified columns exist in the dataset. Optionally
    checks that the DataFrame contains exactly the expected number of
    columns from the baseline (intended for copies that don't expand
    columns). Also verifies that no missing values are present.
    """
    assert df.shape[0] > 0, "Dataset must contain at least one row."
    assert all(col in df.columns for col in columns)
    if check:
        assert df.shape[1] == len(columns)
    missing_values_test(df)


def genre_test(df: pd.DataFrame) -> None:
    """
    Validate that all genres in the dataset belong to the approved set.
    Checks whether every value in the 'genre' column is contained in
    the predefined list of valid genres, raises an AssertionError if not.
    """
    assert df["genre"].isin(top_genres).all(), \
        "Unexpected genres remain in dataset."


def range_sanity_test(df: pd.DataFrame) -> None:
    """
    Validate that scaled values fall within the expected [0, 1] range.
    Ensures that all values in the DataFrame are within the bounds
    produced by MinMax scaling, allowing a small tolerance for floating
    point precision errors.
    """
    assert (df <= 1 + 1e-9).all().all(), \
        "Values above 1 found after scaling."
    assert (df >= -1e-9).all().all(), \
        "Values below 0 found after scaling."


