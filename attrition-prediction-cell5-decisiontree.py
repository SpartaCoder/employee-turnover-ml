# Cell 5: Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred_dtree = dtree.predict(X_val)
print("Decision Tree Validation Accuracy:", accuracy_score(y_val, y_pred_dtree))
print("Classification Report:\n", classification_report(y_val, y_pred_dtree))

cm_dtree = confusion_matrix(y_val, y_pred_dtree)
disp_dtree = ConfusionMatrixDisplay(confusion_matrix=cm_dtree, display_labels=['No', 'Yes'])
disp_dtree.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Decision Tree')
plt.show()

cv_scores_dtree = cross_val_score(dtree, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_dtree)
print("Mean CV:", np.mean(cv_scores_dtree))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_dtree, vert=False)
plt.title("10-Fold CV Accuracy: Decision Tree")
plt.xlabel("Accuracy")
plt.show()
