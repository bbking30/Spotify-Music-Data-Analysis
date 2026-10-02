# Spotify Audio Feature Analysis — CSE 163 Final Project

## Author
Brooks Kahsai

### Project Overview

This project explores relationships between Spotify audio features and track popularity using exploratory data analysis, visualization, statistical testing, and machine learning models. The workflow includes dataset cleaning, feature engineering, genre filtering, statistical summaries, regression analysis, and predictive modeling.

### Project Structure

The analysis is split into modular Python scripts:

* setup.py: Loads and cleans the raw datasets, standardizes column names, performs merging, feature engineering, and creates the final cleaned dataset used across all research questions.
* Q1.py: Performs exploratory regression analysis and visualization of audio features vs standardized popularity.
* Q2.py: Generates statistical summaries and distribution-based analysis of key audio features.
* Q3.py: Applies MinMax scaling and visualizes genre-level feature differences using heatmaps.
* Q4.py: Builds and evaluates machine learning models (Lasso, Random Forest, and MLP) to predict track popularity.
* test.py: Contains assertion-based validation tests used to verify dataset integrity, column consistency, missing-value handling, and scaling bounds.
* eda.py: Runs a 7-number summary exploratory data analysis on the cleaned dataframe from setup.py with the variables of interest for the analysis

### How to Run the Project

1. Install Dependencies 
    
   Make sure you have Python 3.9+ installed, then install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```
2. Dataset Setup 

   Ensure the following CSV files are in the same directory as setup.py:
   * dataset.csv 
   * dataset2.csv 
   * dataset3.csv 
   
   No additional preprocessing is required—the setup.py script handles all cleaning and merging automatically.

3. Run the Analysis 
   
   Run scripts in the following order:

   ```bash
   python setup.py
   python Q1.py
   python Q2.py
   python Q3.py
   python Q4.py
   ```
   
### Notes on Design

* The dataset is cleaned and standardized once in setup.py and reused across all analyses.
* All python files and csv files must be in the same directory level to work properly
* Assertion-based tests are used as checkpoint validations to ensure dataset consistency across transformations.
* Visualization is used extensively for validating trends and feature relationships.
* Machine learning models are evaluated using R² and MSE metrics.
* Visualizations will pop up as extension windows instead of automatically saving to your device in case you don't want your downloads folder flooded with result graphs; pop up window will give you the option to download if you so choose.
   








