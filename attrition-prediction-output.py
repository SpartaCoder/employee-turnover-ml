# ==========================
# Employee Attrition Prediction Output Script
# ==========================

# --- Identify the top models and save their predictions ---

# Find the ML Model with the highest accuracy, mean cv accuracy, and precision
top_accuracy_model = model_metrics_df.loc[model_metrics_df['accuracy'].idxmax(), 'ML Model']
top_cv_model = model_metrics_df.loc[model_metrics_df['mean cv accuracy'].idxmax(), 'ML Model']
top_precision_model = model_metrics_df.loc[model_metrics_df['precision'].idxmax(), 'ML Model']

# Store models to process (avoid duplicates)
models_to_save = set([top_accuracy_model, top_cv_model, top_precision_model])

for model_name in models_to_save:
    if model_name == "Logistic Regression":
        LogicRegressionOutput.to_csv("LogicRegressionOutput.csv", index=False)
    elif model_name == "K-Nearest Neighbors":
        K_NearestNeighborOutput.to_csv("K_NearestNeighborOutput.csv", index=False)
    elif model_name == "Decision Tree":
        DecisionTreeOutput.to_csv("DecisionTreeOutput.csv", index=False)
    elif model_name == "Random Forest":
        RandomForestPredictionOutput.to_csv("RandomForestPredictionOutput.csv", index=False)
    elif model_name == "Naive Bayes":
        NaiveBayesOutput.to_csv("NaiveBayesOutput.csv", index=False)
