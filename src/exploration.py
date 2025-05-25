import pandas as pd

# Load data
data_path = '../data/go_emotions_dataset.csv'
df = pd.read_csv(data_path)

# Quick look
print(df.head())
print(df.info())
print(df['labels'].value_counts())  # Or explore how labels are formatted

# Check for missing data
print(df.isnull().sum())
