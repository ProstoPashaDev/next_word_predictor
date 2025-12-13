from tensorflow.keras.preprocessing.text import Tokenizer


def return_test_dataset():
    text = "I love deep learning. Deep learning is amazing. I love artificial intelligence. AI is the future."
    text = prepare_text(text)

    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1  # +1 for padding
    return text


def prepare_text(text: str):
    return text.lower()


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_seq(text):
    sequences = []
    for line in text.split('.'):
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            sequences.append(tokens[:i+1])

    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len)

    X = sequences[:, :-1]
    y = sequences[:, -1]

