# Tiny Shakespeare GPT

A character-level GPT-style language model trained from scratch on the complete works of William Shakespeare. This project was built to demonstrate a fundamental understanding of the Transformer architecture, tokenization, and the training loop behind modern LLMs.

**This is a learning project, not a production-ready model.** It generates Shakespeare-esque text but has no true understanding of language.
## Features

-   **Built from Scratch:** No Hugging Face `transformers` library, no pre-trained models. Just pure PyTorch.
-   **Custom Character-Level Tokenizer:** A simple but effective tokenizer that maps each unique character to an integer.
-   **GPT-style Transformer Architecture:** Implements core components like:
    -   Token and Positional Embeddings
    -   Multi-Head Self-Attention with Causal Masking
    -   Residual Connections and Layer Normalization
    -   Feed-Forward Blocks
-   **Interactive Chat Interface:** A simple `chat.py` script to interact with the trained model directly from your terminal.

## Architecture

The model is a decoder-only Transformer, similar in spirit to GPT-2. It processes a sequence of characters and autoregressively predicts the next character in the sequence.

-   **Vocabulary Size:** `[Your Vocab Size, e.g., 65 or 100]`
-   **Context Length:** `[Your Context Length, e.g., 128]`
-   **Embedding Dimension:** `[Your Embed Dim, e.g., 128]`
-   **Number of Layers:** `[Your n_layers, e.g., 4]`
-   **Number of Heads:** `[Your n_heads, e.g., 4]`
-   **Total Parameters:** `[Your Parameter Count, e.g., ~800k]`

## How to Use

### 1. Prerequisites

-   Python 3.8+
-   PyTorch
-   (Optional) A sense of humor for the model's creative gibberish.

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone this repository
git clone https://github.com/Yash-Tripath1/Shakespeare-GPT.git
cd Shakespeare-GPT

