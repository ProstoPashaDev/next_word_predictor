import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from src.dataset.data_repository import get_data

# =========================
# Load data
# =========================
train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/test_data.txt"
eval_file  = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/eval.txt"

train_text = get_data(train_file)
eval_text  = get_data(eval_file)

# =========================
# Tokenizer (FIT ONLY ON TRAIN DATA)
# =========================
tokenizer = Tokenizer()
tokenizer.fit_on_texts([train_text])

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# =========================
# Function to create sequences
# =========================
def create_sequences(text, tokenizer):
    sequences = []
    for line in text.split("\n"):
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            sequences.append(tokens[:i + 1])
    return sequences

# =========================
# Create sequences
# =========================
train_sequences = create_sequences(train_text, tokenizer)
eval_sequences  = create_sequences(eval_text, tokenizer)

# =========================
# Padding
# =========================
max_len = max(len(seq) for seq in train_sequences)

train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding="pre")
eval_sequences  = pad_sequences(eval_sequences, maxlen=max_len, padding="pre")

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

X_eval = eval_sequences[:, :-1]
y_eval = eval_sequences[:, -1]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)
y_eval  = tf.keras.utils.to_categorical(y_eval,  num_classes=vocab_size)

# =========================
# Build RNN model
# =========================
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

# =========================
# Train
# =========================
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_eval, y_eval)
)

# =========================
# Evaluate (explicit)
# =========================
loss, acc = model.evaluate(X_eval, y_eval)
print(f"Evaluation loss: {loss:.4f}, accuracy: {acc:.4f}")

# =========================
# Prediction function
# =========================
def predict_next_word(text, model, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
    preds = model.predict(seq, verbose=0)
    return tokenizer.index_word[np.argmax(preds)]

# Example prediction
print(predict_next_word("the cat", model, tokenizer, max_len))
