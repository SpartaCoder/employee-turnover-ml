# Cell 2b: Remove "Employee Count" and "Standard Hours" columns due to no correlation with Attrition

# Documenting removal: These columns are removed because their correlation with Attrition is zero or near-zero,
# meaning they provide no predictive value for the models.

cols_to_remove = []
for col in ["Employee Count", "Standard Hours"]:
    if col in balanced_train.columns:
        cols_to_remove.append(col)

if cols_to_remove:
    print(f"Removing columns due to no correlation with Attrition: {cols_to_remove}")
    balanced_train = balanced_train.drop(columns=cols_to_remove)

# Update features for modeling
columns_to_drop = ['EmployeeNumber', 'Attrition'] + cols_to_remove if 'EmployeeNumber' in balanced_train.columns else ['Attrition'] + cols_to_remove
X = pd.get_dummies(balanced_train.drop(columns=columns_to_drop))
y = balanced_train['Attrition']
