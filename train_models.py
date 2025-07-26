# train_models.py
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load dataset
df = pd.read_csv("ticket_data.csv")

# Combine subject and content
df["text"] = df["subject"].fillna('') + " " + df["content"].fillna('')

# Prepare tokenizer
MAX_WORDS = 5000
MAXLEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=MAXLEN)

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer_json, f)

# Encode labels
module_encoder = LabelEncoder()
request_type_encoder = LabelEncoder()

y_module = to_categorical(module_encoder.fit_transform(df["module_label"]))
y_request = to_categorical(request_type_encoder.fit_transform(df["request_type"]))

# Save the label names if needed later
module_labels = list(module_encoder.classes_)
request_labels = list(request_type_encoder.classes_)

# Simple model builder
def build_model(output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAXLEN))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train and save module model
module_model = build_model(y_module.shape[1])
module_model.fit(X, y_module, epochs=10, batch_size=32, verbose=1)
module_model.save("module_model.h5")

# Train and save request_type model
request_model = build_model(y_request.shape[1])
request_model.fit(X, y_request, epochs=10, batch_size=32, verbose=1)
request_model.save("request_type_model.h5")
