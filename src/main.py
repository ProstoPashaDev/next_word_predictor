import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM

from src.dataset.data_repository import get_data
from src.service.train_settings import StopOnLossThreshold

# =========================
# Load data
# =========================
simple_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_train.txt"
simple_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_eval.txt"

dial_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dialog_train_string_split.txt"
dial_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dialog_eval_string_split.txt"

train_text = get_data(simple_train_file)
#train_text += "\n" + get_data(dial_train_file, num_rows=0)
eval_text = get_data(simple_eval_file)
#eval_text += "\n" + get_data(dial_eval_file, num_rows=0)

# =========================
# Train SentencePiece tokenizer (BPE)
# =========================
SPECIAL_TOKENS = ["<s>", "<user>", "<bot>", "<eos>"]

spm.SentencePieceTrainer.train(
    input=simple_train_file,
    model_prefix="chat_spm",
    vocab_size=500,
    model_type="bpe",
    user_defined_symbols=SPECIAL_TOKENS
)

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file="chat_spm.model")


# =========================
# Helper functions
# =========================
def text_to_sequence(text):
    return sp.encode(text, out_type=int)


def sequence_to_text(ids):
    return sp.decode(ids)


def id_to_word(token_id):
    return sp.id_to_piece(token_id)


# =========================
# Create n-gram sequences
# =========================
def create_sequences(text):
    sequences = []
    for line in text.split("\n"):
        tokens = text_to_sequence(line)
        for i in range(1, len(tokens)):
            sequences.append(tokens[:i + 1])
    return sequences


train_sequences = create_sequences(train_text)
eval_sequences = create_sequences(eval_text)

# =========================
# Padding
# =========================
max_len = max(len(seq) for seq in train_sequences)

train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding="pre")
eval_sequences = pad_sequences(eval_sequences, maxlen=max_len, padding="pre")

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

X_eval = eval_sequences[:, :-1]
y_eval = eval_sequences[:, -1]

vocab_size = sp.get_piece_size()
y_train = tf.keras.utils.to_categorical(y_train, vocab_size)
y_eval = tf.keras.utils.to_categorical(y_eval, vocab_size)

# =========================
# Build RNN model
# =========================
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len - 1),
    LSTM(128),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
stop_on_loss = StopOnLossThreshold(threshold=1)

# =========================
# Train
# =========================
model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_eval, y_eval),
    callbacks=[stop_on_loss]
)


# =========================
# Text generation (sampling)
# =========================
def generate_reply(prompt, max_tokens=30, temperature=0.4):
    for _ in range(max_tokens):
        seq = text_to_sequence(prompt)
        seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")

        preds = model.predict(seq, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        probs = np.exp(preds) / np.sum(np.exp(preds))

        next_id = np.random.choice(len(probs), p=probs)
        next_word = id_to_word(next_id)

        prompt += " " + next_word

        if next_word == "<eos>":
            break

    # Remove prefix and return only bot reply
    bot_reply = prompt.split("<bot>")[-1]
    return bot_reply.strip()


# =========================
# Chat loop
# =========================
print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    prompt = f"<user> {user_input} <bot>"
    response = generate_reply(prompt)

    print("Bot:", end="")
    for word in response.split(" "):
        word = word.replace("▁", " ")
        print(word, end="")
    print()