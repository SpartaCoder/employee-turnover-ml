# Cell 4: K-Nearest Neighbors

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Define features (X) and labels (y) from train_balanced ---
# Adjust 'target_column_name' to your actual target column
target_column = 'Attrition'  # replace with your actual target column name if different
X = train_balanced.drop(columns=[target_column])
y = train_balanced[target_column]

# --- Split into train and test sets (80% train, 20% test) ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train KNN Classifier ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Predict and Evaluate on Validation Set ---
y_pred_knn = knn.predict(X_val)
print("KNN Validation Accuracy:", accuracy_score(y_val, y_pred_knn))
print("Classification Report:\n", classification_report(y_val, y_pred_knn))

# --- Confusion Matrix ---
cm_knn = confusion_matrix(y_val, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['No', 'Yes'])
disp_knn.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: KNN')
plt.show()

# --- 10-Fold Cross Validation ---
cv_scores_knn = cross_val_score(knn, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_knn)
print("Mean CV:", np.mean(cv_scores_knn))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_knn, vert=False)
plt.title("10-Fold CV Accuracy: KNN")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) ---
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_knn_num = le.transform(y_pred_knn)
mae_knn = mean_absolute_error(y_val_num, y_pred_knn_num)
rmae_knn = np.sqrt(mae_knn)
print("Root Mean Absolute Error (RMAE):", rmae_knn)

# --- Compute Metrics for model_metrics_df ---
TN, FP, FN, TP = cm_knn.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Append results to model_metrics_df using pd.concat ---
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

# (Optional) Save if needed
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
