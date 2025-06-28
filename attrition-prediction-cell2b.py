# Cell 2b: Copy all records from train where Attrition == "Yes" into train_balanced
# and produce a visual of the count of number of records by column.

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Select all Attrition == "Yes" records
train_balanced = train[train['Attrition'] == 'Yes'].copy()

# Step 2: Visualize the count of number of records by column in model_train_balanced
record_counts = train_balanced.count()

plt.figure(figsize=(10, 6))
record_counts.plot(kind='bar')
plt.title('Count of Non-Null Records by Column in model_train_balanced')
plt.xlabel('Column')
plt.ylabel('Record Count')
plt.tight_layout()
plt.show()
