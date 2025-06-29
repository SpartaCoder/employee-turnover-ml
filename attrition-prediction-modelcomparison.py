# Cell: Visualize Model Metrics Table
import pandas as pd

# Assuming model_metrics_df is already loaded in your notebook

print("## Model Comparison Table")

# Style the dataframe for better readability
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
display(styled_df)
