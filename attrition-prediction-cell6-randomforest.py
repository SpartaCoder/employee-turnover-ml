# ================================================
# Cell 6: Random Forest Classifier - Training, Evaluation, and Metrics Logging
# Purpose:
#   - Train a Random Forest classifier to predict employee attrition
#   - Evaluate performance using multiple metrics and visualizations
#   - Store results for comparison with other machine learning models
# ================================================
# --- Prepare feature matrix (X) and target vector (y) ---
# Using 'train_unbalanced' DataFrame. Swap to 'train_balanced' if desired.
target_column = 'Attrition'  # Change if your target column name is different
X = train_unbalanced.drop(columns=[target_column])
y = train_unbalanced[target_column]

# --- Split data: 80% training, 20% validation, stratified to preserve class balance ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Random Forest model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- Predict and evaluate on the validation set ---
y_pred_rf = rf.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Classification Report:\n", classification_report(y_val, y_pred_rf))

# --- Display the confusion matrix for validation results ---
cm_rf = confusion_matrix(y_val, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No', 'Yes'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Random Forest')
plt.show()

# --- Perform 10-fold cross-validation for model robustness ---
cv_scores_rf = cross_val_score(rf, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_rf)
print("Mean CV:", np.mean(cv_scores_rf))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_rf, vert=False)
plt.title("10-Fold CV Accuracy: Random Forest")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) on validation set ---
# Encode string labels for numerical calculation
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_rf_num = le.transform(y_pred_rf)
mae_rf = mean_absolute_error(y_val_num, y_pred_rf_num)
rmae_rf = np.sqrt(mae_rf)
print("Root Mean Absolute Error (RMAE):", rmae_rf)

# --- Compute additional metrics for summary table ---
# Unpack confusion matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_rf.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare and append results to the metrics DataFrame for model comparison ---
new_metrics = {
    "ML Model": "Random Forest",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": rmae_rf,
    "mean cv accuracy": np.mean(cv_scores_rf)
}

model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the metrics DataFrame for later analysis ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)

# --- Load the test DataFrame (assumed loaded as 'test' from attrition-prediction-cell1.py) ---
# If not already loaded, import or load it here. Example: from attrition-prediction-cell1 import test

# --- Predict Attrition on the test set using the trained Random Forest model ---
RandomForestPredictionOutput = rf_test.copy()
# Get predicted probabilities for each class (assumes binary: 0 = No, 1 = Yes)
probabilities = rf.predict_proba(rf_test)
RandomForestPredictionOutput['Attrition_Prediction'] = rf.predict(rf_test)
RandomForestPredictionOutput['Probability_No'] = probabilities[:, 0]  # Probability of 'No' attrition
RandomForestPredictionOutput['Probability_Yes'] = probabilities[:, 1]  # Probability of 'Yes' attrition
