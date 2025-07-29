import pandas as pd

# Load the data
df = pd.read_csv('loan_approval_dataset.csv')
print("Column names:")
print(df.columns.tolist())