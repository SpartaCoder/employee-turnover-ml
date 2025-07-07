# Cell 1b: Bar charts showing % of records with Attrition = "Yes" by each column in the dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

for col in train.columns:
    if col == 'Attrition':
        continue
    col_data = train[col]
    # If integer column with >10 unique values, bin into 10 ranges
    if pd.api.types.is_integer_dtype(col_data) and col_data.nunique() > 10:
        binned = pd.cut(col_data, bins=10)
        groupby_col = binned
        plot_type = 'attrition'
    # If non-integer with ≤30 unique values, group by value for attrition rate
    elif not pd.api.types.is_integer_dtype(col_data) and col_data.nunique() <= 30:
        groupby_col = col_data
        plot_type = 'attrition'
    # If non-integer with >30 unique values, plot count of unique values
    elif not pd.api.types.is_integer_dtype(col_data) and col_data.nunique() > 30:
        value_counts = col_data.value_counts().sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        value_counts.plot(kind='bar', color='orange')
        plt.ylabel('Count')
        plt.title(f'Value Counts for {col} (Top {len(value_counts)})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        continue
    else:
        # Integer columns with ≤10 unique values also group by value for attrition rate
        groupby_col = col_data
        plot_type = 'attrition'

    if plot_type == 'attrition':
        attrition_rate = (
            train.groupby(groupby_col)['Attrition']
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_index()
        )
        plt.figure(figsize=(8, 5))
        attrition_rate.plot(kind='bar', color='teal')
        plt.ylabel('% Attrition = "Yes"')
        plt.title(f'Attrition Rate by {col}')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
