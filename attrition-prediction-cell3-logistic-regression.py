# ================================================
# Cell 3: Logistic Regression - Model Training, Evaluation, and Metrics Collection
# Purpose:
#   - Train a logistic regression model for attrition prediction
#   - Evaluate model performance using multiple metrics
#   - Store results for comparison with other models
# ================================================
# --- Prepare features (X) and target (y) ---
# The model uses the 'train_unbalanced' DataFrame with relevant features and the target column 'Attrition'.
X = train_unbalanced.drop('Attrition', axis=1)
y = train_unbalanced['Attrition']

# --- Split the data into training and test sets (80% train, 20% test) ---
# Stratify ensures the class distribution is similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Logistic Regression model ---
# max_iter is raised to 3000 to avoid convergence warnings on difficult data.
logreg = LogisticRegression(max_iter=3000, random_state=42)
logreg.fit(X_train, y_train)

# --- Make predictions on the test set ---
y_pred = logreg.predict(X_test)

# --- Print accuracy and detailed classification report ---
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Display the confusion matrix visually ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Logistic Regression (Test Set)')
plt.show()

# --- Perform 10-fold cross-validation and display accuracy results ---
cv_scores = cross_val_score(logreg, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores, vert=False)
plt.title("10-Fold CV Accuracy: Logistic Regression")
plt.xlabel("Accuracy")
plt.show()

# --- Encode the labels for error calculation ---
# LabelEncoder ensures string labels are converted to integers for MAE calculation.
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
y_pred_num = le.transform(y_pred)  # Use transform to match the same mapping

# --- Calculate and print Root Mean Absolute Error (RMAE) ---
mae = mean_absolute_error(y_test_num, y_pred_num)
rmae = np.sqrt(mae)
print("Root Mean Absolute Error (RMAE):", rmae)

# --- Compute additional model metrics for reporting and comparison ---
# Extract confusion matrix components for metric calculations
# cm is [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare results dictionary for this model ---
new_metrics = {
    "ML Model": "Logistic Regression",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": np.sqrt(mae),
    "mean cv accuracy": np.mean(cv_scores)
}

# --- Append these metrics to the central model_metrics_df DataFrame for comparison ---
model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the updated metrics DataFrame to disk for later use ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)

# --- Predict Attrition on the Test DataFrame from Cell 1 and Store Results ---
# Ensure 'test' has the same features/columns as X_train (may require preprocessing)
LogisticRegressionOutput = test.copy()
LogisticRegressionOutput['Attrition_Prediction'] = logreg.predict(test[X_train.columns])
