'''1. IMDB Reviews – Sentiment Classification
1.1. Overview
The IMDB dataset contains 50 000 movie reviews (≈25 000 for training, 25 000 for validation) labeled positive or negative.
Two typical pipelines:

Approach	Why?	Pros	Cons
LSTM	Classic sequence model.	Simple, fast to train, requires no external libs.	Limited context (short‑term dependencies), slower convergence on long sequences.
BERT	Pre‑trained transformer with contextual embeddings.	State‑of‑the‑art performance on many NLP tasks.	Requires GPU, larger memory, more code.
Below we provide a stand‑alone example for each approach.

# 1.2. LSTM Pipeline (TensorFlow / Keras)

# imdb_lstm.py
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, layers

# 1. Load data
max_features = 20000          # Top 20k words
maxlen = 200                  # Pad/trim reviews to 200 tokens

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features)

# 2. Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_val   = pad_sequences(x_val,   maxlen=maxlen, padding='post')

# 3. Build model
class SentimentLSTM(Model):
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=128):
        super().__init__()
        self.embed = layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)
        self.lstm  = layers.LSTM(lstm_units, return_sequences=False)
        self.drop  = layers.Dropout(0.5)
        self.out   = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embed(x)
        x = self.lstm(x)
        x = self.drop(x)
        return self.out(x)

model = SentimentLSTM(vocab_size=max_features)

# 4. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_data=(x_val, y_val))

# 6. Evaluate
loss, acc = model.evaluate(x_val, y_val)
print(f'Val loss: {loss:.4f}  |  Val acc: {acc:.4f}')
'''
What to tweak

Parameter	Default	Suggested change
max_features	20 000	Raise to 50 000 if memory allows, or lower for speed.
maxlen	200	300–400 for richer context.
embedding_dim	128	256–512 for more capacity.
lstm_units	128	256–512.
epochs	5	10–15; add early‑stopping.
dropout	0.5	0.3–0.6.
'''