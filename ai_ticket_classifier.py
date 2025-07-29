import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional

# === 1. Predefined Classes ===

module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}

request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

# === 2. Load CSV ===

df = pd.read_csv("ticket_data.csv")

# === 3. Combine and Clean Text ===

df['text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'\d+', '', text)  # remove digits
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip().lower()

df['text'] = df['text'].apply(clean_text)

# === 4. Encode Labels ===

df['module_label_enc'] = df['module_label'].map(module_to_id)
df['request_type_enc'] = df['request_type'].map(request_to_id)
df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)

# === 5. Tokenize and Pad Sequences ===

tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, padding='post', maxlen=300)

# === 6. Split Data ===

X_train, X_test, y_mod_train, y_mod_test, y_req_train, y_req_test = train_test_split(
    padded,
    df['module_label_enc'],
    df['request_type_enc'],
    test_size=0.2,
    random_state=42
)

# === 7. Build Model ===

input_layer = Input(shape=(300,))
embedding = Embedding(input_dim=15000, output_dim=64)(input_layer)
x = Bidirectional(LSTM(64, return_sequences=False))(embedding)
x = Dropout(0.5)(x)

module_output = Dense(len(module_labels), activation='softmax', name='module_output')(x)
request_output = Dense(len(request_labels), activation='softmax', name='request_output')(x)

model = Model(inputs=input_layer, outputs=[module_output, request_output])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics={'module_output': 'accuracy', 'request_output': 'accuracy'}
)

model.summary()

# === 8. Train Model ===

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train,
    {'module_output': y_mod_train, 'request_output': y_req_train},
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[early_stop]
)

# === 9. Evaluate ===

results = model.evaluate(X_test, {'module_output': y_mod_test, 'request_output': y_req_test})
print(f"\nModule Accuracy: {results[3]*100:.2f}%")
print(f"Request Type Accuracy: {results[4]*100:.2f}%")

# === 10. Prediction Function ===

def predict_with_summary(subject, content):
    text = clean_text(subject + " " + content)
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=300, padding='post')
    mod_pred, req_pred = model.predict(pad_seq, verbose=0)

    top_mod_indices = mod_pred[0].argsort()[-3:][::-1]
    top_req_indices = req_pred[0].argsort()[-3:][::-1]

    top_mods = [(id_to_module[i], float(mod_pred[0][i])) for i in top_mod_indices]
    top_reqs = [(id_to_request[i], float(req_pred[0][i])) for i in top_req_indices]

    print("\n=== Prediction Summary ===")
    print("Subject:", subject)
    print("Content:", content[:150], "...")

    print("\nTop 3 Predicted Modules:")
    for label, prob in top_mods:
        print(f"  - {label}: {prob*100:.2f}%")

    print("\nTop 3 Predicted Request Types:")
    for label, prob in top_reqs:
        print(f"  - {label}: {prob*100:.2f}%")

    return top_mods[0][0], top_reqs[0][0]

# === 11. Example Test ===

test_subject = "New Employee ID Creation Request"
test_content = "Please create a new employee ID for Mr. Ramesh Kumar who joined on 25th July. Assign to Mumbai cost center 2023."

module, req_type = predict_with_summary(test_subject, test_content)
