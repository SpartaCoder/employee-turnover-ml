# Cell 2: Prepare features, labels, and split data with balanced Attrition

# Map Attrition to binary
train['Attrition'] = train['Attrition'].map({'Yes': 1, 'No': 0})

# Create balanced sample (50% "Yes", 50% "No")
yes_df = train[train['Attrition'] == 1]
no_df = train[train['Attrition'] == 0]
min_count = min(len(yes_df), len(no_df))

# Sample min_count from each group to ensure balance
yes_sample = yes_df.sample(n=min_count, random_state=42)
no_sample = no_df.sample(n=min_count, random_state=42)

# Combine and shuffle
balanced_train = pd.concat([yes_sample, no_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

columns_to_drop = ['EmployeeNumber', 'Attrition'] if 'EmployeeNumber' in balanced_train.columns else ['Attrition']
X = pd.get_dummies(balanced_train.drop(columns=columns_to_drop))
y = balanced_train['Attrition']

# Prepare test data with aligned columns (optional, for future use)
X_test = pd.get_dummies(test.drop(columns=['EmployeeNumber']) if 'EmployeeNumber' in test.columns else test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
