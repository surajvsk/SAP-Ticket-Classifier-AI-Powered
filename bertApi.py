from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import numpy as np
from transformers import BertTokenizer, pipeline
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import re
import json

# ----------- Define Labels -----------
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}
request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

# ----------- Load Tokenizer & Models -----------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = tf.keras.models.load_model("bert_model.h5", custom_objects={"TFBertModel": tf.keras.Model})

# Use distilBART for summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ----------- Helper Functions -----------
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def predict_ticket(subject, content, top_n=3):
    text = clean_text(subject + ' ' + content)
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')

    pred_mod, pred_req = bert_model.predict({
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }, verbose=0)

    top_mod_idx = np.argsort(pred_mod[0])[::-1][:top_n]
    top_req_idx = np.argsort(pred_req[0])[::-1][:top_n]

    module_preds = [(id_to_module[i], float(pred_mod[0][i])) for i in top_mod_idx]
    request_preds = [(id_to_request[i], float(pred_req[0][i])) for i in top_req_idx]

    input_text = (subject + " " + content).strip()
    summary_result = summarizer(input_text[:1024], max_length=60, min_length=15, do_sample=False)
    summary_text = summary_result[0]['summary_text']

    return {
        "subject": subject,
        "content": content[:200] + ("..." if len(content) > 200 else ""),
        "top_module_predictions": [
            {"label": label, "confidence": f"{conf*100:.2f}%"} for label, conf in module_preds
        ],
        "top_request_predictions": [
            {"label": label, "confidence": f"{conf*100:.2f}%"} for label, conf in request_preds
        ],
        "final_prediction": {
            "module": module_preds[0][0],
            "request_type": request_preds[0][0],
            "summary": summary_text
        }
    }

# ----------- Define FastAPI App -----------
app = FastAPI()

class TicketInput(BaseModel):
    subject: str
    content: str

@app.post("/predict")
async def classify_ticket(ticket: TicketInput):
    result = predict_ticket(ticket.subject, ticket.content)
    return result

@app.get("/")
def home():
    return {"message": "Ticket Classification API is running."}
