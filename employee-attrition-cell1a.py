# Cell 1a: Bar chart showing % of records with Attrition = "Yes" by Department

# Calculate % Attrition = Yes by Department
dept_attrition = (
    train.groupby('Department')['Attrition']
    .apply(lambda x: (x == "Yes").mean() * 100)
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
