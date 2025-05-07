import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from transformers import BertTokenizer, BertModel
import torch
import joblib

df = pd.read_csv("grammar_data.csv")  # must have 'text' and 'score'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = bert(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

X = np.array([embed(t) for t in df['text']])
y = df['score']

model = Ridge()
model.fit(X, y)

joblib.dump(model, "grammar/grammar_model.pkl")
