import os

# Directories to search
directories = [
    r"C:\Users\cn230\OneDrive\Desktop",
    r"C:\Users\cn230\Documents",
    r"C:\Users\cn230\Downloads"
]

# Search for train.csv and test.csv
for directory in directories:
    for root, dirs, files in os.walk(directory):
        if "train.csv" in files:
            print(f"✅ Found train.csv at: {os.path.join(root, 'train.csv')}")
        if "test.csv" in files:
            print(f"✅ Found test.csv at: {os.path.join(root, 'test.csv')}")

print("🔍 Search complete!")


def tokenize_function(examples):
    return tokenizer(examples["review_text"], padding="max_length", truncation=True)

import pandas as pd

# Load train.csv
df = pd.read_csv(r"C:\Users\cn230\OneDrive\Desktop\Final Project\train.csv")

# Print column names
print("Column Names:", df.columns)

def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)  # Change "text" to "input"

import pandas as pd

# Load dataset
dataset_path = r"C:\Users\cn230\OneDrive\Desktop\Final Project"

# Clean train.csv
train_df = pd.read_csv(f"{dataset_path}\\train.csv")
train_df = train_df[['input', 'output']]  # Keep only relevant columns
train_df.rename(columns={"input": "text", "output": "label"}, inplace=True)
train_df.to_csv(f"{dataset_path}\\train.csv", index=False)

# Clean test.csv
test_df = pd.read_csv(f"{dataset_path}\\test.csv")
test_df = test_df[['input', 'output']]
test_df.rename(columns={"input": "text", "output": "label"}, inplace=True)
test_df.to_csv(f"{dataset_path}\\test.csv", index=False)

print("✅ Dataset cleaned and saved!")


