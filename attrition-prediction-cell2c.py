# Cell 2c: Bar chart showing % of records with Attrition = "Yes" by Department

# Make sure to use the current training data (after any column removals)
# If using balanced_train or data_for_model, adjust the variable accordingly
department_col = 'Department'
attrition_col = 'Attrition'

# Confirm the column still exists after preprocessing
if department_col in balanced_train.columns:
    dept_attrition = (
        balanced_train.groupby(department_col)[attrition_col]
        .apply(lambda x: (x == 1).mean() * 100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    dept_attrition.plot(kind='bar', color='teal')
    plt.ylabel('% Attrition = "Yes"')
    plt.title('Attrition Rate by Department')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"Column '{department_col}' not found in training data.")
