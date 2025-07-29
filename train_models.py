import pandas as pd
import re
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

# === 1. Labels and mappings ===
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
request_to_id = {label: idx for idx, label in enumerate(request_labels)}

# === 2. Load and clean data ===
df = pd.read_csv("ticket_data.csv")
df['text'] = (df['subject'].fillna('') + ' ' + df['content'].fillna('')).str.lower()
df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
df['module_label_enc'] = df['module_label'].map(module_to_id)
df['request_type_enc'] = df['request_type'].map(request_to_id)
df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)

# === 3. Tokenize using Hugging Face ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(texts, tokenizer, max_len=256):
    return tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')

X_tokenized = tokenize_data(df['text'], tokenizer)
input_ids_np = X_tokenized['input_ids'].numpy()
attention_mask_np = X_tokenized['attention_mask'].numpy()

y_module_np = df['module_label_enc'].values
y_request_np = df['request_type_enc'].values

# === 4. Train/Test Split ===
input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, y_mod_train, y_mod_test, y_req_train, y_req_test = train_test_split(
    input_ids_np, attention_mask_np, y_module_np, y_request_np, test_size=0.2, random_state=42
)

# === 5. Build model with Hugging Face BERT ===
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = Input(shape=(256,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(256,), dtype=tf.int32, name='attention_mask')

def get_bert_output(inputs):
    ids, mask = inputs
    bert_output = bert_model(ids, attention_mask=mask)[0]  # shape: (batch, seq_len, hidden)
    return bert_output[:, 0, :]  # CLS token (first token)

bert_output = Lambda(get_bert_output, output_shape=(768,))([input_ids, attention_mask])

module_output = Dense(len(module_labels), activation='softmax', name='module_output')(bert_output)
request_output = Dense(len(request_labels), activation='softmax', name='request_output')(bert_output)

model = Model(inputs=[input_ids, attention_mask], outputs=[module_output, request_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics={'module_output': 'accuracy', 'request_output': 'accuracy'}
)

# === 6. Train ===
history = model.fit(
    {'input_ids': input_ids_train, 'attention_mask': attention_mask_train},
    {'module_output': y_mod_train, 'request_output': y_req_train},
    validation_split=0.1,
    epochs=3,
    batch_size=8
)

# === 7. Save Model ===
model.save("multi_output_bert_model.keras")

# === 8. Save Tokenizer ===
tokenizer.save_pretrained("bert_tokenizer")

print("✅ Hugging Face model and tokenizer saved successfully.")
