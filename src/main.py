import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional

from src.dataset.data_repository import get_data
from src.service.train_settings import StopOnLossThreshold

# =========================
# Parameters tuning
# =========================

LSTM_UNITS = 128
EMDEB = 64
VOCABULARY_SIZE = 2000

# =========================
# Load data
# =========================
simple_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_train.txt"
simple_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_eval.txt"

dial_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dialog_train_string_split.txt"
dial_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/dialog_eval_string_split.txt"

diff_quest_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/different_questions_train.txt"
diff_quest_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/different_questions_eval.txt"

word_meaning_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/word_meaning_train.txt"
word_meaning_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/word_meaning_eval.txt"

simple_dialog_train_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_dialog_train.txt"
simple_dialog_eval_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/simple_dialog_eval.txt"


train_text = "\n".join([f for f in [
        get_data(simple_train_file),
        #get_data(diff_quest_train_file, num_rows=100),
        #get_data(word_meaning_train_file, num_rows=100)
        #get_data(dial_train_file, num_rows=100)
        get_data(simple_dialog_train_file)
]])

print(train_text)

eval_text = "\n".join([f for f in [
    get_data(simple_eval_file),
    #get_data(diff_quest_eval_file, num_rows=10),
    #get_data(word_meaning_eval_file, num_rows=10)
    #get_data(dial_eval_file, num_rows=100)
    get_data(simple_dialog_eval_file)
]])

print()
print("========================================")
print()

print(eval_text)

# =========================
# Train SentencePiece tokenizer (BPE)
# =========================
SPECIAL_TOKENS = ["<s>", "<user>", "<bot>", "<eos>"]

spm_input_file = "C:/KhramovPavel/Project/Python/NextWordPredictor/recources/spm_input.txt"

with open(spm_input_file, "w", encoding="utf-8") as f:
    f.write(train_text)

spm.SentencePieceTrainer.train(
    input=spm_input_file,
    model_prefix="chat_spm",
    vocab_size=VOCABULARY_SIZE,
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
    Embedding(vocab_size, EMDEB, input_length=max_len - 1),
    Bidirectional(LSTM(LSTM_UNITS)),
    #LSTM(LSTM_UNITS),
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
    callbacks=[stop_on_loss, early_stop]
)


# =========================
# Text generation (sampling)
# =========================
def generate_reply(prompt, max_tokens=30, temperature=0.7):
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