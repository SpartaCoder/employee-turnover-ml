# Cell 7: Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_val)
print("Naive Bayes Validation Accuracy:", accuracy_score(y_val, y_pred_nb))
print("Classification Report:\n", classification_report(y_val, y_pred_nb))

cm_nb = confusion_matrix(y_val, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['No', 'Yes'])
disp_nb.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Naive Bayes')
plt.show()

cv_scores_nb = cross_val_score(nb, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_nb)
print("Mean CV:", np.mean(cv_scores_nb))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_nb, vert=False)
plt.title("10-Fold CV Accuracy: Naive Bayes")
plt.xlabel("Accuracy")
plt.show()
