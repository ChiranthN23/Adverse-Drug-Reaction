from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model
model_path = "saved_model"  # Ensure this folder contains your trained model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Example text for prediction
text = "The patient has a severe fever."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

print("Predicted Class:", predicted_class)
