# Economic Dialogue Generation Chatbot

This project implements a neural network-based chatbot designed to assist users in understanding fundamental economic concepts and providing financial explanations. The chatbot leverages Bidirectional LSTM networks, SentencePiece tokenization, and top-k sampling to generate coherent responses in economic dialogues.

Features:

- Token-level and subword-level modeling of economic text.
- Bidirectional LSTM architecture to capture both past and future context.
- Top-k sampling for controlled, coherent text generation.
- Early stopping and dropout regularization for improved generalization.
- Support for integrating pretrained embeddings (e.g., GloVe).
- Interactive chat interface with <user> and <bot> roles.

Requirements:

- Python 3.9+
- TensorFlow 2.x
- Numpy
- Matplotlib
- SentencePiece
- (Optional) Pretrained GloVe embeddings

Installation:

1. Clone the repository:

   git clone https://github.com/your-repo/economic-chatbot.git
   cd economic-chatbot

2. Install dependencies:

   pip install -r requirements.txt

3. Prepare dataset files:

   - simple_train.txt, simple_eval.txt – general dialogue data  
   - economic_train.txt, economic_eval.txt – economic domain dialogue data  

   Place them under `recources/`.

Training:

1. Preprocess data and train SentencePiece tokenizer:

   python train_chatbot.py

2. The model uses the following hyperparameters:

   - LSTM units: 192
   - Embedding dimension: 128
   - Vocabulary size: 2500
   - Dropout: 0.2
   - Recurrent dropout: 0.2
   - Batch size: 32
   - Epochs: 20
   - Temperature for sampling: 0.4

3. Early stopping is applied based on validation loss to prevent overfitting.

Usage:

1. Run the chatbot:

   python train_chatbot.py

2. Interact with the bot in the console. Type "quit" to exit.

3. Example interaction:

   You: What is inflation?
   Bot: Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.

Notes:

- Top-k sampling is used during text generation to restrict token selection to the most probable candidates, improving coherence.
- Pretrained embeddings can be incorporated by uncommenting the embedding section in the code.
- The chatbot can handle multi-turn dialogues with `<user>` and `<bot>` tokens.
