# ================================================
# Cell 2: Compute and Visualize Correlation Matrix
# Purpose: Identify features with little or no correlation to Attrition
# Dependencies: Uses only matplotlib (no seaborn needed)
# ================================================
# --- Convert Attrition to numeric if necessary ---
# This ensures the Attrition column is numeric for correlation analysis:
# 'Yes' becomes 1, 'No' becomes 0.
if train['Attrition'].dtype == object:
    train['Attrition_numeric'] = train['Attrition'].map({'Yes': 1, 'No': 0})
    target_col = 'Attrition_numeric'
else:
    target_col = 'Attrition'

# --- Compute correlation matrix (numeric columns only) ---
cor_matrix = train.corr(numeric_only=True)

# --- Extract correlation of each feature with Attrition ---
# Remove self-correlation and sort features by absolute correlation value (strongest to weakest)
corr_with_attrition = (
    cor_matrix[target_col]
    .drop(target_col)
    .sort_values(key=abs, ascending=False)
)

print("Correlation of features with Attrition:")
print(corr_with_attrition)

# --- Identify features with very low correlation (absolute value < 0.01) ---
no_corr_features = corr_with_attrition[abs(corr_with_attrition) < 0.01].index.tolist()
print("\nFeatures with near-zero correlation to Attrition (|corr| < 0.01):")
print(no_corr_features)

# --- Visualize the entire correlation matrix as a heatmap ---
plt.figure(figsize=(12, 8))
plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(cor_matrix.columns)), cor_matrix.columns, rotation=90)
plt.yticks(range(len(cor_matrix.index)), cor_matrix.index)
plt.title('Correlation Matrix (All Features)')
plt.tight_layout()
plt.show()

# --- Visualize correlations with Attrition as a horizontal bar chart ---
plt.figure(figsize=(4, len(corr_with_attrition) * 0.4))
plt.barh(corr_with_attrition.index, corr_with_attrition.values)
plt.title('Feature Correlation with Attrition')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
