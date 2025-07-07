# ================================================
# Cell 4: K-Nearest Neighbors (KNN) Model Training, Evaluation, and Metrics Logging
# Purpose:
#   - Train a KNN classifier for attrition prediction
#   - Assess performance using various evaluation metrics and plots
#   - Store results for comparison with other models in a metrics DataFrame
# ================================================
# --- Prepare feature matrix (X) and target vector (y) ---
# You may swap between 'train_balanced' or 'train_unbalanced' as needed.
target_column = 'Attrition'
X = train_unbalanced.drop(columns=[target_column])
y = train_unbalanced[target_column]

# --- Split data: 80% training, 20% validation, stratified to maintain class ratio ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the KNN model (default k=5) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Predict and evaluate on the validation set ---
y_pred_knn = knn.predict(X_val)
print("KNN Validation Accuracy:", accuracy_score(y_val, y_pred_knn))
print("Classification Report:\n", classification_report(y_val, y_pred_knn))

# --- Display the confusion matrix for validation results ---
cm_knn = confusion_matrix(y_val, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['No', 'Yes'])
disp_knn.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: KNN')
plt.show()

# --- Perform 10-fold cross-validation for model robustness ---
cv_scores_knn = cross_val_score(knn, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_knn)
print("Mean CV:", np.mean(cv_scores_knn))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_knn, vert=False)
plt.title("10-Fold CV Accuracy: KNN")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) on validation set ---
# LabelEncoder ensures consistent encoding for string labels.
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_knn_num = le.transform(y_pred_knn)
mae_knn = mean_absolute_error(y_val_num, y_pred_knn_num)
rmae_knn = np.sqrt(mae_knn)
print("Root Mean Absolute Error (RMAE):", rmae_knn)

# --- Compute standard metrics for summary table ---
# Unpack confusion matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_knn.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare and append results to the metrics DataFrame ---
new_metrics = {
    "ML Model": "K-Nearest Neighbors",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": rmae_knn,
    "mean cv accuracy": np.mean(cv_scores_knn)
}

model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the metrics DataFrame for future reference ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
