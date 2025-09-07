# Decoder-Only Transformer

This repository contains a simple implementation of a decoder-only transformer model, built from scratch using PyTorch. The primary goal of this project is to provide a clear and concise example of how such a model works, with a particular focus on the Query, Key, and Value (QKV) calculations in the multi-head self-attention mechanism.

## Architecture

The model is a standard decoder-only transformer, similar to the architecture of GPT models. It consists of the following components:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds positional information to the token embeddings.
- **Decoder Blocks**: A stack of decoder blocks, each containing:
    - **Multi-Head Self-Attention**: Allows the model to weigh the importance of different tokens in the input sequence. This is where the Q, K, and V calculations happen.
    - **Feed-Forward Network**: A simple fully connected neural network applied to each position.
- **Final Linear Layer**: Maps the output of the decoder blocks to the vocabulary size to produce logits.
- **Softmax**: Converts logits to probabilities.

## Usage

To train the model and generate text, you first need to install the dependencies:

```bash
pip install -r requirements.txt
```

Then, you can run the model:

```bash
python model.py
```

The script will train the model on the text in `input.txt` and then generate a new text sequence.

## Q, K, V Calculations

The core of the self-attention mechanism is the calculation of Query, Key, and Value vectors. For each input token, we create a Query, a Key, and a Value vector.

- **Query**: Represents the current token's "question" about other tokens.
- **Key**: Represents what "information" each token has.
- **Value**: Represents the actual content of each token.

The attention score between two tokens is calculated by taking the dot product of the Query vector of the current token and the Key vector of the other token. These scores are then scaled, and a softmax is applied to get the attention weights. Finally, the Value vectors are multiplied by the attention weights and summed up to produce the output for the current token. This process is done in parallel for multiple "heads," and the results are concatenated and projected to get the final output of the multi-head attention layer.
