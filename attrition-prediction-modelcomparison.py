# ================================================
# Cell: Visualize Model Metrics Table
# Purpose:
#   - Display a comparative summary of model metrics collected during training and evaluation
#   - Use pandas styling for enhanced readability and visual insight
#   - Assumes 'model_metrics_df' is already defined and contains metrics for each model
# ================================================
# Print heading for clarity in notebook output or logs
print("## Model Comparison Table")

# --- Style the DataFrame for improved readability and insight ---
# - Set a table caption for context
# - Format all numeric columns to 3 decimal places for consistency
# - Highlight 'accuracy' and 'mean cv accuracy' metrics using a blue gradient for quick visual comparison
# - Hide the index for a cleaner look
styled_df = (
    model_metrics_df.style
    .set_caption("Model Metrics Comparison")
    .format({
        "accuracy": "{:.3f}",
        "specificity": "{:.3f}",
        "sensitivity": "{:.3f}",
        "precision": "{:.3f}",
        "root mean absolute error": "{:.3f}",
        "mean cv accuracy": "{:.3f}",
    })
    .background_gradient(cmap="Blues", subset=["accuracy", "mean cv accuracy"])
    .hide_index()
)

# --- Display the styled DataFrame (Jupyter/IPython only) ---
display(styled_df)
