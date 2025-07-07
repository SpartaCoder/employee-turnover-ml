# ==========================
# Employee Attrition Prediction Output Script
# This script:
#   - Loads the training and test datasets
#   - Trains a Random Forest classifier on the training data
#   - Applies the trained model to the test data
#   - Outputs a CSV with the required "index" and "Output" ("Yes"/"No") columns
# ==========================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load Datasets ---
# Assumes 'train.csv' and 'test.csv' are in the same directory as this script.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# --- 2. Prepare Training Data ---
target_column = 'Attrition'  # Target variable indicating attrition ("Yes"/"No")

# Features (X): All columns except target
X_train = train.drop(columns=[target_column])
# Target (y): Attrition column
y_train = train[target_column]

# --- 3. Prepare Test Data ---
# The test data may include an 'index' column for identification. We'll keep it for the output.
test_index = test['index'] if 'index' in test.columns else test.index

# For prediction, drop 'index' if present
X_test = test.drop(columns=['index']) if 'index' in test.columns else test.copy()

# --- 4. Train Random Forest Model ---
# Note: For reproducibility, the random_state is fixed.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- 5. Make Predictions on Test Data ---
# The model predicts "Yes"/"No" strings directly if the target column was not label-encoded.
y_pred = rf.predict(X_test)

# --- 6. Prepare Output DataFrame ---
output_df = pd.DataFrame({
    'index': test_index,   # Copy original test index or 'index' column
    'Output': y_pred       # Model's prediction ("Yes" or "No")
})

# --- 7. Save Output ---
output_df.to_csv('attrition_rf_output.csv', index=False)
print("Output file 'attrition_rf_output.csv' created successfully.")
