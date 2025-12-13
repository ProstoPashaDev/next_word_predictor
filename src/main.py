import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load text
with open("C:/KhramovPavel/Project/Python/NextWordPredictor/recources/test_data.txt", "r") as f:
    text = f.read().lower()

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Create seqs
sequences = []

for line in text.split("\n"):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        sequences.append(tokens[:i + 1])

# Padding
max_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

X = sequences[:, :-1]
y = sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build base RNN model

model = Sequential([
    Embedding(vocab_size, 50, input_length=max_len - 1),
    SimpleRNN(128),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# Epochs

model.fit(
    X, y,
    epochs=50,
    batch_size=16
)


def predict_next_word(text, model, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
    preds = model.predict(seq, verbose=0)
    return tokenizer.index_word[np.argmax(preds)]


print(predict_next_word("Kirill", model, tokenizer, max_len))
