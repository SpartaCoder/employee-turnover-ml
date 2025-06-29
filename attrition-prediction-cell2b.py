# Cell 2b: Balance dataset by copying selected columns from train where Attrition == "Yes",
# add equal number of random "No" records, visualize feature missingness by Attrition as a stacked bar chart,
# and copy selected columns from train to train_unbalanced.

import pandas as pd
import matplotlib.pyplot as plt

# Assuming cor_df is available from attrition-prediction-cell2a.py and contains "To Remove" column
cols_to_keep = cor_df[cor_df["To Remove"] == False].index.tolist()

# Ensure 'Attrition' is included for filtering and plotting
if 'Attrition' not in cols_to_keep:
    cols_to_keep.append('Attrition')

# Step 1: Select all records with Attrition == "Yes" and only the relevant columns
yes_records = train.loc[train['Attrition'] == 'Yes', cols_to_keep].copy()

# Step 2: Select an equal number of random records with Attrition == "No"
num_yes = len(yes_records)
no_candidates = train.loc[train['Attrition'] == 'No', cols_to_keep]
no_records = no_candidates.sample(n=num_yes, random_state=42).copy()  # random_state for reproducibility

# Step 3: Concatenate both sets to create a balanced dataframe
train_balanced = pd.concat([yes_records, no_records], axis=0).reset_index(drop=True)

# Step 4: Create train_unbalanced with the same selected columns (but unbalanced)
#         Only the columns where "To Remove" is False from corr_df are kept.
train_unbalanced = train[cols_to_keep].copy()

# Step 5: Compute counts of non-null values for each feature, broken down by Attrition value
# Exclude 'Attrition' itself from the features to plot
features = [col for col in cols_to_keep if col != 'Attrition']
counts_by_attrition = (
    train_balanced.groupby('Attrition')[features]
    .apply(lambda df: df.notnull().sum())
    .T  # Transpose for plotting (features on x-axis)
)

# Step 6: Plot as a stacked bar chart
counts_by_attrition.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    color=['#1f77b4', '#ff7f0e'],
    edgecolor='black'
)
plt.title('Non-Null Record Count by Feature (Stacked by Attrition)')
plt.xlabel('Feature')
plt.ylabel('Non-Null Record Count')
plt.legend(title='Attrition')
plt.tight_layout()
plt.show()
