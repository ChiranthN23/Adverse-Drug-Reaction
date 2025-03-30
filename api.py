from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Initialize FastAPI app
app = FastAPI()

# Load Model & Tokenizer
MODEL_PATH = "saved_model"  # Change this if you have a different path
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Request Body Structure
class TextInput(BaseModel):
    text: str

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    label = "ADR (Adverse Drug Reaction)" if predicted_class == 1 else "Non-ADR"
    return {"text": text, "predicted_class": label, "confidence": round(confidence, 2)}

# API Endpoint
@app.post("/predict")
def classify_text(input_data: TextInput):
    return predict(input_data.text)


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running successfully!"}

