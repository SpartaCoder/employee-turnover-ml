# Cell 3: Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_val)
print("Logistic Regression Validation Accuracy:", accuracy_score(y_val, y_pred_logreg))
print("Classification Report:\n", classification_report(y_val, y_pred_logreg))

cm_logreg = confusion_matrix(y_val, y_pred_logreg)
disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=['No', 'Yes'])
disp_logreg.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Logistic Regression')
plt.show()

cv_scores_logreg = cross_val_score(logreg, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_logreg)
print("Mean CV:", np.mean(cv_scores_logreg))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_logreg, vert=False)
plt.title("10-Fold CV Accuracy: Logistic Regression")
plt.xlabel("Accuracy")
plt.show()
