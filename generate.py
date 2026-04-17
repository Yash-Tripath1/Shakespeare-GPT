# generate.py

import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import CharTokenizer

# --- Configuration ---
MODEL_PATH = "best_model.pt"
TOKENIZER_PATH = "tokenizer.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_and_tokenizer():
    """Loads the trained model and tokenizer from disk."""
    print("Loading tokenizer...")
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    
    print("Loading model checkpoint...")
    # map_location ensures it loads correctly even if trained on a different device
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Re-create the model using the same config it was trained with
    model = GPT(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval() # Set the model to evaluation mode (disables dropout, etc.)
    
    print("✓ Model and tokenizer loaded successfully.")
    return model, tokenizer

@torch.no_grad() # We don't need to calculate gradients for inference
def generate(model, tokenizer, seed_text, max_new_tokens, temperature=1.0):
    """
    Generates text autoregressively from a seed string.
    """
    # 1. Encode the seed text and prepare it as a tensor
    # We add an extra dimension at the beginning to represent the batch size of 1
    context = torch.tensor(tokenizer.encode(seed_text), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # 2. Loop to generate one token at a time
    for _ in range(max_new_tokens):
        # Crop the context to the model's maximum context length
        # This is important because our model can't process sequences longer than it was trained on
        context_cropped = context[:, -model.context_length:]
        
        # 3. Get the model's predictions (logits) for the next token
        logits, _ = model(context_cropped)
        
        # We only care about the very last token's prediction in the sequence
        logits = logits[:, -1, :] # Shape becomes (Batch=1, Vocab_Size)
        
        # 4. Apply temperature to the logits
        logits = logits / temperature
        
        # 5. Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        
        # 6. Sample the next token from the probability distribution
        # multinomial samples one index based on the probabilities
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 7. Append the new token to our context for the next iteration
        context = torch.cat((context, next_token), dim=1)
        
    # 8. Decode the entire generated sequence back to text
    return tokenizer.decode(context[0].tolist())


if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n--- Generating Text ---")
    
    # Try a few different seeds and temperatures!
    seed = "the night felt long "
    print(f"Seed: '{seed}', Temperature: 0.8")
    
    generated_text = generate(
        model, 
        tokenizer, 
        seed_text=seed, 
        max_new_tokens=500, 
        temperature=0.8
    )
    
    print("\nGenerated Output:\n")
    print(generated_text)
    print("\n--- End of Generation ---")