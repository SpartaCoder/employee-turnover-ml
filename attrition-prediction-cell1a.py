# ================================================
# Cell 1a: Null Value Check, Imputation, and Visualization
# ================================================
import pandas as pd
import numpy as np

# Function to check and impute nulls, and generate a summary table
def nulls_imputation_summary(df, df_name="DataFrame"):
    summary = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        non_null_count = df[col].notnull().sum()
        percent_null = (null_count / len(df)) * 100
        
        if null_count > 0:
            # Determine type for imputation
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric: use mean
                impute_value = df[col].mean()
                impute_type = "mean"
            else:
                # Categorical: use mode
                impute_value = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                impute_type = "mode"
            # Impute missing values
            df[col].fillna(impute_value, inplace=True)
        else:
            impute_value = np.nan
            impute_type = None
        
        # Always get the mean and mode for reporting
        col_mean = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else np.nan
        col_mode = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        
        # Only append rows with original nulls for the table
        if null_count > 0:
            summary.append({
                'column': col,
                'null_count': null_count,
                'non_null_count': non_null_count,
                'percent_null': percent_null,
                'mean_used_for_imputation': col_mean,
                'mode_used_for_imputation': col_mode
            })
    # Convert summary to DataFrame for pretty display
    summary_df = pd.DataFrame(summary, columns=[
        'column', 'null_count', 'non_null_count', 'percent_null',
        'mean_used_for_imputation', 'mode_used_for_imputation'
    ])
    print(f"\nNull Value Imputation Summary for {df_name}:")
    if not summary_df.empty:
        display(summary_df)
    else:
        print("No null values found.")

# Check and impute nulls in train and test, and visualize
nulls_imputation_summary(train, df_name="train")
nulls_imputation_summary(test, df_name="test")
