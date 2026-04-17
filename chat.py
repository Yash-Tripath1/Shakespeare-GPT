# chat.py
import torch
from model import GPT
from tokenizer import CharTokenizer

# --- Configuration ---
DEVICE = 'cpu'
MODEL_CHECKPOINT = 'best_model.pt'  # Load our new storyteller model
TOKENIZER_CHECKPOINT = 'tokenizer.json'

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
tokenizer = CharTokenizer.load(TOKENIZER_CHECKPOINT)
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model = GPT(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()
print("✓ Ready to chat!")

# --- The Generation Logic (from generate.py) ---
@torch.no_grad()
def generate(seed_text, max_new_tokens=100, temperature=0.7):
    # (Copy the exact 'generate' function from your generate.py here)
    # ... but let's write it again for clarity
    context = torch.tensor(tokenizer.encode(seed_text), dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        context_cond = context[:, -model.context_length:]
        logits, _ = model(context_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)
    return tokenizer.decode(context[0].tolist())

# --- The Interactive Chat Loop ---
print("\n--- Tiny Storyteller GPT ---")
print("Enter your prompt or type 'quit' to exit.")
while True:
    prompt = input("\n> ")
    if prompt.lower() == 'quit':
        break
    
    # Generate the story based on the prompt
    output = generate(prompt)
    
    # Print the full story
    print(output)