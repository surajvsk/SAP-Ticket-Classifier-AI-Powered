from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn

import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split



# ===================== SETUP ======================

app = FastAPI()

# === 1. Labels ===

module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}
request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

# === 2. Clean Text ===

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# === 3. Load and Prepare Data ===

df = pd.read_csv("ticket_data.csv")  # Replace with actual path
df['text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')
df['text'] = df['text'].apply(clean_text)

df['module_label_enc'] = df['module_label'].map(module_to_id)
df['request_type_enc'] = df['request_type'].map(request_to_id)
df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, padding='post', maxlen=300)

X_train, X_test, y_mod_train, y_mod_test, y_req_train, y_req_test = train_test_split(
    padded,
    df['module_label_enc'],
    df['request_type_enc'],
    test_size=0.2,
    random_state=42
)

# === 4. Build and Train Model ===

input_layer = Input(shape=(300,))
embedding = Embedding(input_dim=10000, output_dim=64)(input_layer)
x = LSTM(64)(embedding)

module_output = Dense(len(module_labels), activation='softmax', name='module_output')(x)
request_output = Dense(len(request_labels), activation='softmax', name='request_output')(x)

model = Model(inputs=input_layer, outputs=[module_output, request_output])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics={'module_output': 'accuracy', 'request_output': 'accuracy'}
)

model.fit(
    X_train,
    {'module_output': y_mod_train, 'request_output': y_req_train},
    validation_split=0.2,
    epochs=10,
    batch_size=8
)


# === 5. Add Simple Summary Function ===

def generate_summary(text: str, max_sentences: int = 2) -> str:
    """
    Naive extractive summary: Picks first N meaningful sentences.
    """
    # Break into sentences using simple punctuation (can use nltk for better accuracy)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return ' '.join(sentences[:max_sentences])


# ===================== FASTAPI ======================

class TicketInput(BaseModel):
    subject: str
    content: str

@app.post("/predict")
def predict(input_data: TicketInput):
    text = clean_text(input_data.subject + " " + input_data.content)
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=300, padding='post')

    mod_pred, req_pred = model.predict(pad_seq)

    top_mod_indices = mod_pred[0].argsort()[-3:][::-1]
    top_req_indices = req_pred[0].argsort()[-3:][::-1]

    top_modules = [{"label": id_to_module[i], "probability": float(mod_pred[0][i])} for i in top_mod_indices]
    top_requests = [{"label": id_to_request[i], "probability": float(req_pred[0][i])} for i in top_req_indices]

    # Generate summary from original input
    full_text = input_data.subject + " " + input_data.content
    summary = generate_summary(full_text)

    return {
        "subject": input_data.subject,
        "content": input_data.content[:150],
        "top_modules": top_modules,
        "top_request_types": top_requests,
        "predicted_module": top_modules[0]["label"],
        "predicted_request_type": top_requests[0]["label"],
        "summery": summary
    }

    
# Optional to run locally with: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)