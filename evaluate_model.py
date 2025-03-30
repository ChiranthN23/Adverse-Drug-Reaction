from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load the trained model and tokenizer
MODEL_PATH = "saved_model"  # Update this if needed
TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

# Set model to evaluation mode
model.eval()

def classify_text(text, threshold=0.60):  # Lowered threshold to 0.60
    """
    Classifies input text as ADR or Non-ADR based on the trained model.
    
    Args:
        text (str): The input text to classify.
        threshold (float): Confidence threshold for ADR classification.
    
    Returns:
        str: Classification result (ADR or Non-ADR).
        float: Confidence score of ADR class.
    """
    # Tokenize the input with truncation and padding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1)  # Convert logits to probabilities

    confidence = probs[0][1].item()  # Confidence for ADR class
    predicted_class = 1 if confidence > threshold else 0
    category = "ADR (Adverse Drug Reaction)" if predicted_class == 1 else "Non-ADR"

    return category, confidence

# Example inputs
examples = [
    "The patient developed a severe rash after taking the medication.",
    "The patient shows no signs of infection and is stable.",
    "In a patient suffering from rheumatoid arthritis, we report the first simultaneous occurrence of two side effects of low-dose methotrexate: an acute megaloblastic anaemia and a pneumonitis.",
    "No side effects were observed during the treatment period."
]

# Run classification on examples
for text in examples:
    category, confidence = classify_text(text)
    print(f"Input: {text}")
    print(f"Predicted Class: {category}")
    print(f"Confidence: {confidence:.2f}\n")
    
    
from transformers import BertForSequenceClassification, BertTokenizer

# Define the path where you saved the model
MODEL_PATH = "saved_model"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")
