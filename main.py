from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import traceback
# === Custom Lambda Function (must match what was used in training) ===
@tf.keras.utils.register_keras_serializable()
def get_bert_output(inputs):
    input_ids, attention_mask = inputs
    bert_model = tf.keras.models.load_model("bert_encoder_model.keras")
    return bert_model([input_ids, attention_mask])[0][:, 0, :]  # CLS token output

# === Labels ===
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_type_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']
MAXLEN = 256

# === Load Tokenizer and Model ===
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = load_model("multi_output_bert_model.keras", compile=True)

# === FastAPI App ===
app = FastAPI(title="SAP Ticket Classifier using BERT")

class Ticket(BaseModel):
    subject: str
    content: str

@app.post("/predict")
def predict(ticket: Ticket):
    try:
        text = f"{ticket.subject} {ticket.content}".lower()
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAXLEN,
            return_tensors='tf'
        )

        outputs = model.predict(
            {
                "input_ids": encoding['input_ids'],
                "attention_mask": encoding['attention_mask']
            },
            verbose=0
        )

        module_probs, request_probs = outputs
        predicted_module = module_labels[np.argmax(module_probs[0])]
        predicted_request_type = request_type_labels[np.argmax(request_probs[0])]

        return {
            "module": predicted_module,
            "request_type": predicted_request_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
