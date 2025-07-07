# ================================================
# Cell 5: Decision Tree Classifier - Training, Evaluation, and Metrics Logging
# Purpose:
#   - Train a Decision Tree classifier for employee attrition prediction
#   - Evaluate its performance using multiple metrics and visualizations
#   - Store results for comparison with other machine learning models
# ================================================
# --- Prepare feature matrix (X) and target vector (y) ---
# This example uses 'train_unbalanced', but you can swap with 'train_balanced' as needed.
target_column = 'Attrition'  # Change this if your target column name is different
X = train_unbalanced.drop(columns=[target_column])
y = train_unbalanced[target_column]

# --- Split data: 80% training, 20% validation, stratified to maintain class ratio ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Decision Tree model ---
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# --- Predict and evaluate on the validation set ---
y_pred_dtree = dtree.predict(X_val)
print("Decision Tree Validation Accuracy:", accuracy_score(y_val, y_pred_dtree))
print("Classification Report:\n", classification_report(y_val, y_pred_dtree))

# --- Display the confusion matrix for validation results ---
cm_dtree = confusion_matrix(y_val, y_pred_dtree)
disp_dtree = ConfusionMatrixDisplay(confusion_matrix=cm_dtree, display_labels=['No', 'Yes'])
disp_dtree.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Decision Tree')
plt.show()

# --- Perform 10-fold cross-validation for model robustness ---
cv_scores_dtree = cross_val_score(dtree, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_dtree)
print("Mean CV:", np.mean(cv_scores_dtree))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_dtree, vert=False)
plt.title("10-Fold CV Accuracy: Decision Tree")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) on validation set ---
# Convert string labels to integers for error calculation
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_dtree_num = le.transform(y_pred_dtree)
mae_dtree = mean_absolute_error(y_val_num, y_pred_dtree_num)
rmae_dtree = np.sqrt(mae_dtree)
print("Root Mean Absolute Error (RMAE):", rmae_dtree)

# --- Compute additional metrics for summary table ---
# Extract confusion matrix components: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_dtree.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare and append results to the metrics DataFrame for comparison ---
new_metrics = {
    "ML Model": "Decision Tree",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": rmae_dtree,
    "mean cv accuracy": np.mean(cv_scores_dtree)
}

model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the metrics DataFrame for later analysis ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)

# --- Predict Attrition on the Test DataFrame from Cell 1 and Store Results ---
# Ensure the test set has the same feature columns as X_train.
DecisionTreeOutput = test.copy()
probs = dtree.predict_proba(test[X_train.columns])
# Assuming 'No' (not leaving) is class 0 and 'Yes' (leaving) is class 1
DecisionTreeOutput['Pass_Probability'] = probs[:, 0]
DecisionTreeOutput['Fail_Probability'] = probs[:, 1]
DecisionTreeOutput['Attrition_Prediction'] = dtree.predict(test[X_train.columns])
