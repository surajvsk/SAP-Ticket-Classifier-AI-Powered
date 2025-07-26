import pandas as pd
import re
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# === 1. Labels and mappings ===
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}

request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

# === 2. Load and clean data ===
df = pd.read_csv("ticket_data.csv")
df['text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

df['text'] = df['text'].apply(clean_text)
df['module_label_enc'] = df['module_label'].map(module_to_id)
df['request_type_enc'] = df['request_type'].map(request_to_id)
df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)

# === 3. Tokenize ===
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, padding='post', maxlen=300)

# === 4. Split data ===
X_train, X_test, y_mod_train, y_mod_test, y_req_train, y_req_test = train_test_split(
    padded, df['module_label_enc'], df['request_type_enc'], test_size=0.2, random_state=42
)

# === 5. Define model ===
input_layer = Input(shape=(300,))
embedding = Embedding(input_dim=10000, output_dim=64)(input_layer)
x = LSTM(64)(embedding)

module_output = Dense(len(module_labels), activation='softmax', name='module_output')(x)
request_output = Dense(len(request_labels), activation='softmax', name='request_output')(x)

model = Model(inputs=input_layer, outputs=[module_output, request_output])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics={'module_output': 'accuracy', 'request_output': 'accuracy'})

# === 6. Train ===
model.fit(X_train, {'module_output': y_mod_train, 'request_output': y_req_train},
          validation_split=0.2, epochs=10, batch_size=8)

# === 7. Save models ===
module_model = Model(inputs=model.input, outputs=model.get_layer('module_output').output)
request_model = Model(inputs=model.input, outputs=model.get_layer('request_output').output)
module_model.save("module_model.h5")
request_model.save("request_type_model.h5")

# === 8. Save tokenizer ===
with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

print("✅ Models and tokenizer saved.")
