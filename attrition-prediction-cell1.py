# ================================================
# Cell 1: Import libraries and load data
# Project: Employee Turnover ML (IBM Kaggle Dataset)
# Description: Prepare environment, import core libraries, and load datasets
# ================================================

# --- Import essential Python libraries ---
import pandas as pd                           # Data manipulation and analysis
import numpy as np                            # Numerical operations
import matplotlib.pyplot as plt               # Data visualization

# --- Import scikit-learn utilities for ML workflow ---
from sklearn.model_selection import train_test_split, cross_val_score  # Data splitting & model validation
from sklearn.metrics import (classification_report,                    # Model evaluation tools
                             accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

# --- Import classifiers to be used ---
from sklearn.linear_model import LogisticRegression         # Logistic Regression classifier
from sklearn.neighbors import KNeighborsClassifier          # K-Nearest Neighbors classifier
from sklearn.tree import DecisionTreeClassifier             # Decision Tree classifier
from sklearn.ensemble import RandomForestClassifier         # Random Forest classifier
from sklearn.naive_bayes import GaussianNB                  # Gaussian Naive Bayes classifier

# --- Load training and test datasets ---
# Assumes 'train.csv' and 'test.csv' are present in the root directory (same as this script)
train = pd.read_csv('train.csv')    # Training data
test = pd.read_csv('test.csv')      # Test data

# The datasets are now loaded as pandas DataFrames: `train` and `test`
# Further data exploration and preprocessing will follow in subsequent cells.
