import whisper
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Load whisper model
whisper_model = whisper.load_model("base")

# Load BERT and Ridge Regressor
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
regressor = joblib.load("grammar/grammar_model.pkl")  # Ensure this model exists

def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def predict_score(text):
    embedding = embed_text(text)
    score = regressor.predict([embedding])[0]
    return round(score, 2)
