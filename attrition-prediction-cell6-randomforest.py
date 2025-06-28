# Cell 6: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Classification Report:\n", classification_report(y_val, y_pred_rf))

cm_rf = confusion_matrix(y_val, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No', 'Yes'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Random Forest')
plt.show()

cv_scores_rf = cross_val_score(rf, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_rf)
print("Mean CV:", np.mean(cv_scores_rf))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_rf, vert=False)
plt.title("10-Fold CV Accuracy: Random Forest")
plt.xlabel("Accuracy")
plt.show()
