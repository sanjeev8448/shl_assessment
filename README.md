# 🎤 Grammar Scoring Engine (Voice-Based) – SHL Internship Project

This is a Django-based web application that allows users to upload audio files (voice samples). The system uses OpenAI's Whisper to transcribe the audio to text and then uses a BERT-based Ridge Regression model to assign a grammar score to the transcribed text.

---

## 🧠 Key Features

- 🎧 Audio Upload via web interface
- 🤖 Automatic transcription using Whisper ASR
- 🔤 BERT-based embedding of transcribed text
- 📊 Grammar score prediction using Ridge Regression
- 💾 Stores audio, transcription, and scores in database

---

## ⚙️ Tech Stack

- **Backend**: Django (Python)
- **AI/ML**:
  - OpenAI Whisper (Speech to Text)
  - BERT (Text Embedding)
  - Ridge Regressor (Grammar Scoring)
- **Frontend**: HTML (Django Templates)
- **Database**: SQLite (default Django)

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/grammar-scoring-engine.git
cd grammar-scoring-engine


grammar_scoring_engine/
│
├── grammar/
│   ├── migrations/
│   ├── templates/
│   │   └── upload.html
│   ├── __init__.py
│   ├── admin.py
│   ├── ai_engine.py     ← Transcription + Scoring
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
├── grammar_scoring_engine/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── media/                ← Uploaded audio files
├── db.sqlite3
├── manage.py
├── requirements.txt
└── grammar_model.pkl     ← Pre-trained Ridge regression model


To train your own grammar_model.pkl:
# train_model.py

import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import Ridge
import torch
import joblib
import numpy as np

df = pd.read_csv("grammar_data.csv")  # contains 'text' and 'score'

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
