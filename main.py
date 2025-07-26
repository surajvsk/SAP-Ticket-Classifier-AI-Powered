from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

# === Load Models ===
module_model = load_model("module_model.h5")
request_type_model = load_model("request_type_model.h5")

# === Load Tokenizer ===
with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    # tokenizer = tokenizer_from_json(tokenizer_data)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))  # Convert dict to JSON string
# === Label Definitions ===
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_type_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']
MAXLEN = 300

# === FastAPI App ===
app = FastAPI(title="SAP Ticket Classifier")

class Ticket(BaseModel):
    subject: str
    content: str

@app.post("/predict")
def predict_ticket(ticket: Ticket):
    try:
        text = f"{ticket.subject} {ticket.content}".lower()
        seq = tokenizer.texts_to_sequences([text])
        if not seq or not any(seq[0]):
            raise HTTPException(status_code=400, detail="Input text contains no recognizable tokens.")

        padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')

        module_pred = module_model.predict(padded, verbose=0)
        request_pred = request_type_model.predict(padded, verbose=0)

        if module_pred.shape[1] != len(module_labels):
            raise HTTPException(status_code=500, detail="Mismatch in module label count and model output.")
        if request_pred.shape[1] != len(request_type_labels):
            raise HTTPException(status_code=500, detail="Mismatch in request type label count and model output.")

        predicted_module = module_labels[np.argmax(module_pred[0])]
        predicted_request_type = request_type_labels[np.argmax(request_pred[0])]

        return {
            "module": predicted_module,
            "request_type": predicted_request_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
