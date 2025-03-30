import pandas as pd
from transformers import AutoTokenizer

# Load dataset
file_path = "C:/Users/cn230/OneDrive/Desktop/Final Project/Python files/adverse_drug_events_trained_datasets.csv"
df = pd.read_csv(file_path)

# Initialize tokenizer (using BioBERT)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
df["tokenized"] = df["input"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=128))

# Save tokenized dataset
df.to_csv("C:/Users/cn230/OneDrive/Desktop/Final Project/Python files/tokenized_dataset.csv", index=False)

print("✅ Tokenization complete. File saved as tokenized_dataset.csv")