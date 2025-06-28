# Cell 2c: ML Model Metrics DataFrame Creation

import pandas as pd

# Define the columns for model evaluation metrics
metrics_columns = [
    "ML Model",
    "accuracy",
    "specificity",
    "sensitivity",
    "precision",
    "root mean absolute error",
    "mean cv accuracy"
]

# Create an empty DataFrame to store results for each ML model
model_metrics_df = pd.DataFrame(columns=metrics_columns)

# Example: To add results for a particular model, use the following code:
# model_metrics_df = model_metrics_df.append({
#     "ML Model": "RandomForest",
#     "accuracy": accuracy_value,
#     "specificity": specificity_value,
#     "sensitivity": sensitivity_value,
#     "precision": precision_value,
#     "root mean absolute error": rmae_value,
#     "mean cv accuracy": mean_cv_accuracy_value
# }, ignore_index=True)

# The DataFrame `model_metrics_df` can then be used to store and display results for each trained model.
