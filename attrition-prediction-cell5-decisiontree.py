# Cell 5: Decision Tree

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Define features (X) and labels (y) from train_balanced ---
# target_column = 'Attrition'  # Change this if your target column name is different
# X = train_balanced.drop(columns=[target_column])
# y = train_balanced[target_column]

# --- Define features (X) and labels (y) from train_unbalanced ---
target_column = 'Attrition'  # Change this if your target column name is different
X = train_unbalanced.drop(columns=[target_column])
y = train_unbalanced[target_column]

# --- Split into train and test sets (80% train, 20% test) ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Decision Tree Classifier ---
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# --- Predict and Evaluate on Validation Set ---
y_pred_dtree = dtree.predict(X_val)
print("Decision Tree Validation Accuracy:", accuracy_score(y_val, y_pred_dtree))
print("Classification Report:\n", classification_report(y_val, y_pred_dtree))

# --- Confusion Matrix ---
cm_dtree = confusion_matrix(y_val, y_pred_dtree)
disp_dtree = ConfusionMatrixDisplay(confusion_matrix=cm_dtree, display_labels=['No', 'Yes'])
disp_dtree.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Decision Tree')
plt.show()

# --- 10-Fold Cross Validation ---
cv_scores_dtree = cross_val_score(dtree, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_dtree)
print("Mean CV:", np.mean(cv_scores_dtree))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_dtree, vert=False)
plt.title("10-Fold CV Accuracy: Decision Tree")
plt.xlabel("Accuracy")
plt.show()

# --- Calculate Root Mean Absolute Error (RMAE) ---
le = LabelEncoder()
y_val_num = le.fit_transform(y_val)
y_pred_dtree_num = le.transform(y_pred_dtree)
mae_dtree = mean_absolute_error(y_val_num, y_pred_dtree_num)
rmae_dtree = np.sqrt(mae_dtree)
print("Root Mean Absolute Error (RMAE):", rmae_dtree)

# --- Computing Metrics for model_metrics_df ---
TN, FP, FN, TP = cm_dtree.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Append results to model_metrics_df using pd.concat ---
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

# (Optional) Save if needed
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
