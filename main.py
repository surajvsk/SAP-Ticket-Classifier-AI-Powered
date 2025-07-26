from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import numpy as np
import json

# Load models
module_model = load_model("module_model.h5")
request_type_model = load_model("request_type_model.h5")

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Categories (used during training)
module_labels = ['FICO', 'HCM', 'MM', 'SD']
request_type_labels = ['CR Modification', 'Incident', 'Service Request']

MAXLEN = 100

# FastAPI app
app = FastAPI(title="SAP Ticket Classifier")

# Request schema
class Ticket(BaseModel):
    subject: str
    content: str

# Prediction endpoint
@app.post("/predict")
def predict_ticket(ticket: Ticket):
    try:
        text = ticket.subject + " " + ticket.content
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAXLEN)

        module_pred = module_model.predict(padded)[0]
        request_pred = request_type_model.predict(padded)[0]

        predicted_module = module_labels[np.argmax(module_pred)]
        predicted_request_type = request_type_labels[np.argmax(request_pred)]

        return {
            "module": predicted_module,
            "request_type": predicted_request_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
