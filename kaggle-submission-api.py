# Import necessary libraries
import pandas as pd
import os

# Load the LogisticRegressionOutput DataFrame
# Replace 'LogisticRegressionOutput.csv' with the actual file path if different
logreg_output = pd.read_csv('LogisticRegressionOutput.csv', index_col=0)

# Prepare the submission file with only 'index' and 'Output' columns
# 'index' is taken from the DataFrame's index, 'Output' is from 'Attrition_Prediction'
submission = pd.DataFrame({
    'index': logreg_output.index,
    'Output': logreg_output['Attrition_Prediction']
})

# Save the submission file as 'submission.csv'
submission.to_csv('submission.csv', index=False)

# Submit the file to Kaggle using the Kaggle API
# Make sure your environment is authenticated with `kaggle` (see https://github.com/Kaggle/kaggle-api)
os.system('kaggle competitions submit -c ai-and-ml-level-1-kaggle-competition -f submission.csv -m "Logistic Regression submission"')
