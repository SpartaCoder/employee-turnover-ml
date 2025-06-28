# Cell 2a: Identify features with weak or NaN correlation with Attrition and visualize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Identify features to remove: correlation is NaN or between -0.1 and 0.1 (exclusive)
corr_mask = corr_with_attrition.isna() | ((corr_with_attrition > -0.1) & (corr_with_attrition < 0.1))
cols_to_remove = corr_with_attrition[corr_mask].index.tolist()

if cols_to_remove:
    print(f"Columns with weak or NaN correlation with Attrition: {cols_to_remove}")

# Create a new dataframe, model_train, that excludes columns to remove
model_train = train.drop(columns=[col for col in cols_to_remove if col in train.columns])

# Table visualization of the correlation matrix, highlighting features to be removed in red font
cor_df = corr_with_attrition.to_frame(name='Correlation')
cor_df['To Remove'] = cor_df.index.isin(cols_to_remove)
styled = cor_df.style.apply(
    lambda col: ['color: red' if rem else '' for rem in cor_df['To Remove']],
    subset=['Correlation']
).format({'Correlation': "{:.2f}"})

from IPython.display import display
display(styled)
