# Cell 3: Logistic Regression

# --- Import required libraries ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Make sure pandas is imported for concat

# --- Prepare features (X) and target (y) from train_balanced DataFrame ---
# X = train_balanced.drop('Attrition', axis=1)
# y = train_balanced['Attrition']

# --- Prepare features (X) and target (y) from train_unbalanced DataFrame ---
X = train_unbalanced.drop('Attrition', axis=1)
y = train_unbalanced['Attrition']

# --- Split data into 80% train and 20% test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Logistic Regression model ---
# Increased max_iter to 3000 to prevent ConvergenceWarning
logreg = LogisticRegression(max_iter=3000, random_state=42)
logreg.fit(X_train, y_train)

# --- Make predictions on the test set ---
y_pred = logreg.predict(X_test)

# --- Print accuracy and classification report ---
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Display the confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Logistic Regression (Test Set)')
plt.show()

# --- Perform 10-fold cross validation and display results ---
cv_scores = cross_val_score(logreg, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores, vert=False)
plt.title("10-Fold CV Accuracy: Logistic Regression")
plt.xlabel("Accuracy")
plt.show()

# --- Encode the labels for MAE calculation to avoid ValueError ---
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
y_pred_num = le.transform(y_pred)  # Use transform, not fit_transform

# --- Calculate and print Root Mean Absolute Error (RMAE) ---
mae = mean_absolute_error(y_test_num, y_pred_num)
rmae = np.sqrt(mae)
print("Root Mean Absolute Error (RMAE):", rmae)

# --- At the end of attrition-prediction-cell3-logistic-regression.py ---
# Assuming 'cm' is a 2x2 matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Prepare the new row as a dictionary
new_metrics = {
    "ML Model": "Logistic Regression",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": np.sqrt(mae),
    "mean cv accuracy": np.mean(cv_scores)
}

# Append to the DataFrame using pd.concat (instead of append)
model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# (Optional) Save the updated DataFrame if needed:
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# OR
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)
