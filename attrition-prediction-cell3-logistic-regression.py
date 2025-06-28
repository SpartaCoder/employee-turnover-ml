# Cell 3: Logistic Regression

# --- Import required libraries ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error
)
import matplotlib.pyplot as plt
import numpy as np

# --- Prepare features (X) and target (y) from train_balanced DataFrame ---
X = train_balanced.drop('Attrition', axis=1)
y = train_balanced['Attrition']

# --- Split data into 80% train and 20% test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize and train the Logistic Regression model ---
logreg = LogisticRegression(max_iter=1000, random_state=42)
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

# --- Calculate and print Root Mean Absolute Error (RMAE) ---
mae = mean_absolute_error(y_test, y_pred)
rmae = np.sqrt(mae)
print("Root Mean Absolute Error (RMAE):", rmae)
