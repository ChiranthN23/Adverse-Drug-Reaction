import os

file_path = "C:/Users/cn230/OneDrive/Desktop/Final Project/Python files/adverse_drug_events_trained_datasets.csv"

if os.path.exists(file_path):
    print(f"✅ File found: {file_path}")
else:
    print(f"❌ File NOT found: {file_path}")