# Cell 4: K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_val)
print("KNN Validation Accuracy:", accuracy_score(y_val, y_pred_knn))
print("Classification Report:\n", classification_report(y_val, y_pred_knn))

cm_knn = confusion_matrix(y_val, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['No', 'Yes'])
disp_knn.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: KNN')
plt.show()

cv_scores_knn = cross_val_score(knn, X, y, cv=10)
print("10-Fold CV Scores:", cv_scores_knn)
print("Mean CV:", np.mean(cv_scores_knn))
plt.figure(figsize=(6, 4))
plt.boxplot(cv_scores_knn, vert=False)
plt.title("10-Fold CV Accuracy: KNN")
plt.xlabel("Accuracy")
plt.show()
