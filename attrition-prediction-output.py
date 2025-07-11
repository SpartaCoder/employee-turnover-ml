# =========================================================
# Employee Attrition Prediction: Output Top Model Results
# =========================================================

# This script identifies the top-performing ML models based on different evaluation metrics
# and saves their prediction outputs to separate CSV files for further analysis.

# --- Identify the best models by evaluation metric ---

# Select the model with the highest accuracy score from the results DataFrame
top_accuracy_model = model_metrics_df.loc[model_metrics_df['accuracy'].idxmax(), 'ML Model']

# Select the model with the best mean cross-validation accuracy
top_cv_model = model_metrics_df.loc[model_metrics_df['mean cv accuracy'].idxmax(), 'ML Model']

# Select the model with the highest precision value
top_precision_model = model_metrics_df.loc[model_metrics_df['precision'].idxmax(), 'ML Model']

# --- Collect unique models to avoid saving duplicates ---

# Use a set to ensure each model is only saved once, even if it excels in multiple metrics
models_to_save = set([top_accuracy_model, top_cv_model, top_precision_model])

# --- Save predictions for each top model to CSV ---

for model_name in models_to_save:
    # Check which model corresponds to the current name and save its predictions
    if model_name == "Logistic Regression":
        # Save Logistic Regression predictions
        LogisticRegressionOutput.to_csv("LogicRegressionOutput.csv", index=True)
    elif model_name == "K-Nearest Neighbors":
        # Save KNN predictions
        K_NearestNeighborOutput.to_csv("K_NearestNeighborOutput.csv", index=True)
    elif model_name == "Decision Tree":
        # Save Decision Tree predictions
        DecisionTreeOutput.to_csv("DecisionTreeOutput.csv", index=True)
    elif model_name == "Random Forest":
        # Save Random Forest predictions
        RandomForestPredictionOutput.to_csv("RandomForestPredictionOutput.csv", index=True)
    elif model_name == "Naive Bayes":
        # Save Naive Bayes predictions
        NaiveBayesOutput.to_csv("NaiveBayesOutput.csv", index=True)
