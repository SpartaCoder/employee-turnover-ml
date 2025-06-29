# Cell 6: Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Train Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- Predict and Evaluate on Validation Set ---
y_pred_rf = rf.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Classification Report:\n", classification_report(y_val, y_pred_rf))

# --- Confusion Matrix ---
cm_rf = confusion_matrix(y_val, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No', 'Yes'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Random Forest')
plt.show()

# --- 10-Fold Cross Validation ---
cv_scores_rf = cross_val_score(rf, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_rf)
print("Mean CV:", np.mean(cv_scores_rf))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_rf, vert=False)
plt.title("10-Fold CV Accuracy: Random Forest")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_rf_num = le.transform(y_pred_rf)
mae_rf = mean_absolute_error(y_val_num, y_pred_rf_num)
rmae_rf = np.sqrt(mae_rf)
print("Root Mean Absolute Error (RMAE):", rmae_rf)

# --- Compute Metrics for model_metrics_df ---
TN, FP, FN, TP = cm_rf.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Append results to model_metrics_df using pd.concat ---
new_metrics = {
    "ML Model": "Rnadom Forest",  # Consider changing to "Random Forest"
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

# (Optional) Save if needed
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
