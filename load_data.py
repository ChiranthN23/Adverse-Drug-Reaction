import pandas as pd

file_path = "C:/Users/cn230/OneDrive/Desktop/Final Project/Python files/adverse_drug_events_trained_datasets.csv"

df = pd.read_csv(file_path)
print(df.head())  # Show first few rows to confirm it's loaded