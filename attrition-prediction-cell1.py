# ================================================
# Cell 1: Import libraries and load data
# Project: Employee Turnover ML (IBM Kaggle Dataset)
# Description: Prepare environment, import core libraries, and load datasets
# ================================================

# --- Import essential Python libraries ---
import pandas as pd                           # Data manipulation and analysis using DataFrames
import numpy as np                            # Numerical operations and array handling
import matplotlib.pyplot as plt               # Data visualization (for charts/plots)

# --- Import scikit-learn utilities for ML workflow ---
from sklearn.preprocessing import LabelEncoder             # Encode categorical variables as numeric
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and validating models
from sklearn.metrics import (classification_report,        # Evaluate model performance with detailed metrics
                             accuracy_score,               # Metric: overall accuracy
                             confusion_matrix,             # Metric: confusion matrix for classification
                             ConfusionMatrixDisplay,       # Visualization for confusion matrix
                             mean_absolute_error)          # Metric: mean absolute error (regression/classification)

# --- Import classifiers to be used ---
from sklearn.linear_model import LogisticRegression         # Baseline classifier: Logistic Regression
from sklearn.neighbors import KNeighborsClassifier          # Instance-based classifier: K-Nearest Neighbors
from sklearn.tree import DecisionTreeClassifier             # Tree-based classifier: Decision Tree
from sklearn.ensemble import RandomForestClassifier         # Ensemble classifier: Random Forest
from sklearn.naive_bayes import GaussianNB                  # Probabilistic classifier: Gaussian Naive Bayes

# --- Load training and test datasets ---
# Assumes 'train.csv' and 'test.csv' are present in the root directory (same as this script)
train = pd.read_csv('train.csv')    # Load training dataset as pandas DataFrame
test = pd.read_csv('test.csv')      # Load test dataset as pandas DataFrame

# The datasets are now loaded as pandas DataFrames: `train` and `test`
# Further data exploration and preprocessing will follow in subsequent cells.
