import pandas as pd
import re
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from transformers import pipeline
import torch
# import torch
# 1. Labels
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}
request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

# 2. Load & Clean Data
df = pd.read_csv("ticket_data.csv")
df['text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

df['text'] = df['text'].apply(clean_text)
df['module_label_enc'] = df['module_label'].map(module_to_id)
df['request_type_enc'] = df['request_type'].map(request_to_id)
df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)

# 3. Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_texts(texts, max_len=128):
    return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors='np')

X = tokenize_texts(df['text'])
input_ids = X['input_ids']
attention_mask = X['attention_mask']
y_module = np.array(df['module_label_enc'])
y_request = np.array(df['request_type_enc'])

# 4. Train-Test Split
input_ids_train, input_ids_val, attn_train, attn_val, y_mod_train, y_mod_val, y_req_train, y_req_val = train_test_split(
    input_ids, attention_mask, y_module, y_request, test_size=0.2, random_state=42
)

train_inputs = {
    'input_ids': input_ids_train,
    'attention_mask': attn_train
}
val_inputs = {
    'input_ids': input_ids_val,
    'attention_mask': attn_val
}

# 5. Build Model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

input_ids_layer = Input(shape=(128,), dtype=tf.int32, name='input_ids')
attn_layer = Input(shape=(128,), dtype=tf.int32, name='attention_mask')

def get_bert_output(inputs):
    input_ids, attn = inputs
    outputs = bert_model(input_ids=input_ids, attention_mask=attn)
    return outputs.pooler_output

bert_output = Lambda(
    get_bert_output,
    output_shape=(768,),
    name='bert_lambda'
)([input_ids_layer, attn_layer])
dropout = Dropout(0.3)(bert_output)

module_output = Dense(len(module_labels), activation='softmax', name='module_output')(dropout)
request_output = Dense(len(request_labels), activation='softmax', name='request_output')(dropout)

model = Model(inputs=[input_ids_layer, attn_layer], outputs=[module_output, request_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=[['accuracy'], ['accuracy']]
)

model.summary()

# 6. Train Model
model.fit(
    train_inputs,
    {'module_output': y_mod_train, 'request_output': y_req_train},
    validation_data=(val_inputs, {'module_output': y_mod_val, 'request_output': y_req_val}),
    epochs=3,
    batch_size=16
)

# Save model architecture and weights
model.save("bert_model.h5")

# Optional: Save tokenizer to reuse later
tokenizer.save_pretrained("tokenizer/")

# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# summarizer = pipeline("summarization", model="Falconsai/text_summarization", framework="tf")

# 7. Predict Function with Summary Output
def predict_ticket(subject, content, top_n=3):
    text = clean_text(subject + ' ' + content)
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')

    pred_mod, pred_req = model.predict({'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}, verbose=0)

    top_mod_idx = np.argsort(pred_mod[0])[::-1][:top_n]
    top_req_idx = np.argsort(pred_req[0])[::-1][:top_n]

    module_preds = [(id_to_module[i], float(pred_mod[0][i])) for i in top_mod_idx]
    request_preds = [(id_to_request[i], float(pred_req[0][i])) for i in top_req_idx]

    # Generate NLP-based summary
    input_text = (subject + " " + content).strip()
    summary_result = summarizer(input_text[:1024], max_length=60, min_length=15, do_sample=False)
    summary_text = summary_result[0]['summary_text']

    summary = {
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
            "summery": summary_text
        }
    }

    return summary

# 8. Test Demo
if __name__ == "__main__":
    subject = "New Employee ID Creation Request"
    content = "Please create a new employee ID for Mr. Ramesh Kumar who joined on 25th July. Assign to Mumbai cost center 2023."

    result = predict_ticket(subject, content)
    import json
    print(json.dumps(result, indent=2))
