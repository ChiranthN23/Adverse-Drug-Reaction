import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Check if dataset exists
dataset_path = "c:/Users/cn230/OneDrive/Desktop/Final Project/dataset.csv"  # Use the full path

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the file path.")

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset Columns:", df.columns)

# Ensure correct column names
if "text" not in df.columns or "label" not in df.columns:
    raise KeyError("Dataset must have 'text' and 'label' columns.")

# Convert labels to integers if needed
df["label"] = df["label"].astype(int)

# Split into train/test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./bert_model")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_function)

# Label encoding
def label_encoding(example):
    return {"labels": torch.tensor(example["label"], dtype=torch.long)}

dataset = dataset.map(label_encoding)

# Split dataset for training/testing
dataset = dataset.train_test_split(test_size=0.2)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(df["label"].unique()),
    cache_dir="./bert_model"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Start training
trainer.train()

