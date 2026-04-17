import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer
from model import GPT

# --- Hyperparameters ---
# These are the same as in model.py, but we define them here to use them
BATCH_SIZE = 32      # How many sequences we process in parallel
CONTEXT_LENGTH = 128 # The maximum length of a sequence
EMBED_DIM = 128      # The size of our token and positional embeddings
N_HEADS = 4          # Number of attention heads
N_LAYERS = 4         # Number of Transformer blocks
DROPOUT = 0.1        # Dropout rate
LEARNING_RATE = 3e-4 # Learning rate for the optimizer

# Training-specific hyperparameters
MAX_ITERS = 5000       # Total number of training iterations
EVAL_INTERVAL = 500    # How often to evaluate and print loss
EVAL_ITERS = 100       # How many batches to use for evaluation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available, else CPU
print(f"Using device: {DEVICE}")
# For your setup, this will print 'cpu'
class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        # The number of possible starting points for a sequence
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # Grab a chunk of text of size context_length + 1
        chunk = self.data[idx:idx + self.context_length + 1]
        
        # The input is the first 'context_length' characters
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # The target is the next 'context_length' characters (shifted by one)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    # --- Main Training Logic ---

# 1. Load data and tokenizer
print("Loading data...")
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer.load("tokenizer.json")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# 2. Split data into training and validation sets
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = TextDataset(train_data, CONTEXT_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Note: We don't need a full val_loader, we can just grab random batches
# for evaluation to save memory and time.

# 3. Initialize model and optimizer
model = GPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    context_length=CONTEXT_LENGTH,
    dropout=DROPOUT
).to(DEVICE)

print(f"Model has {model.count_parameters():,} parameters.")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# A helper function for evaluation
@torch.no_grad() # This tells PyTorch not to calculate gradients, saving memory/compute
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            # Grab a random batch of data
            start_idx = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))
            x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in start_idx])
            y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in start_idx])
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

# 4. The Training Loop
print("Starting training...")
best_val_loss = 1e9 # A very large number
for iter in range(MAX_ITERS):
    # Every once in a while, evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save the model if validation loss is the best we've seen
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            print("✓ New best model saved!")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'embed_dim': EMBED_DIM,
                    'n_layers': N_LAYERS,
                    'n_heads': N_HEADS,
                    'context_length': CONTEXT_LENGTH,
                    'dropout': DROPOUT
                }
            }, 'best_model.pt')


    # Get a batch of data
    # We can't use the DataLoader directly in this simple loop structure
    # So we'll sample batches manually, which is fine for this project
    start_idx = torch.randint(len(train_data) - CONTEXT_LENGTH, (BATCH_SIZE,))
    xb = torch.stack([train_data[i:i+CONTEXT_LENGTH] for i in start_idx]).to(DEVICE)
    yb = torch.stack([train_data[i+1:i+CONTEXT_LENGTH+1] for i in start_idx]).to(DEVICE)

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
    optimizer.step()

print("Training finished.")