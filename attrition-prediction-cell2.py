# Cell 2: Compute and visualize correlation matrix to identify features with no correlation to Attrition
# Uses only matplotlib (no seaborn dependency)

# Compute correlation matrix using only numeric columns
cor_matrix = train.corr(numeric_only=True)

# Correlation of all features with 'Attrition'
corr_with_attrition = cor_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)
print("Correlation of features with Attrition:")
print(corr_with_attrition)

# Show features with low correlation (absolute value < 0.01)
no_corr_features = corr_with_attrition[abs(corr_with_attrition) < 0.01].index.tolist()
print("\nFeatures with near-zero correlation to Attrition (|corr| < 0.01):")
print(no_corr_features)

# Visualize correlation matrix as a heatmap using matplotlib only
plt.figure(figsize=(12, 8))
plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(cor_matrix.columns)), cor_matrix.columns, rotation=90)
plt.yticks(range(len(cor_matrix.index)), cor_matrix.index)
plt.title('Correlation Matrix (All Features)')
plt.tight_layout()
plt.show()

# Optionally, plot only correlations with Attrition as a bar chart
plt.figure(figsize=(4, len(corr_with_attrition) * 0.4))
plt.barh(corr_with_attrition.index, corr_with_attrition.values)
plt.title('Feature Correlation with Attrition')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
