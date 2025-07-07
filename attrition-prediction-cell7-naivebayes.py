# ================================================
# Cell 7: Naive Bayes Classifier - Training, Evaluation, and Metrics Logging
# Purpose:
#   - Train a Naive Bayes classifier for employee attrition prediction
#   - Evaluate its performance using a suite of metrics and visualizations
#   - Store results for comparison with other machine learning models
# ================================================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Prepare feature matrix (X) and target vector (y) ---
# Using 'train_unbalanced' DataFrame. You can swap to 'train_balanced' if desired.
target_column = 'Attrition'  # Change this if your target column name is different
X = train_unbalanced.drop(columns=[target_column])
y = train_unbalanced[target_column]

# --- Split data: 80% training, 20% validation, stratified to preserve class balance ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Naive Bayes model ---
nb = GaussianNB()
nb.fit(X_train, y_train)

# --- Predict and evaluate on the validation set ---
y_pred_nb = nb.predict(X_val)
print("Naive Bayes Validation Accuracy:", accuracy_score(y_val, y_pred_nb))
print("Classification Report:\n", classification_report(y_val, y_pred_nb))

# --- Display the confusion matrix for validation results ---
cm_nb = confusion_matrix(y_val, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['No', 'Yes'])
disp_nb.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Naive Bayes')
plt.show()

# --- Perform 10-fold cross-validation for model robustness ---
cv_scores_nb = cross_val_score(nb, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_nb)
print("Mean CV:", np.mean(cv_scores_nb))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_nb, vert=False)
plt.title("10-Fold CV Accuracy: Naive Bayes")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) on validation set ---
# Encode string labels for numerical calculation
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_nb_num = le.transform(y_pred_nb)
mae_nb = mean_absolute_error(y_val_num, y_pred_nb_num)
rmae_nb = np.sqrt(mae_nb)
print("Root Mean Absolute Error (RMAE):", rmae_nb)

# --- Compute additional metrics for summary table ---
# Unpack confusion matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_nb.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare and append results to the metrics DataFrame for model comparison ---
new_metrics = {
    "ML Model": "Naive Bayes",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": rmae_nb,
    "mean cv accuracy": np.mean(cv_scores_nb)
}

model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the metrics DataFrame for later analysis ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
