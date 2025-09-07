import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the model definition from your existing model.py file
from model import DecoderTransformer, Block, MultiHeadAttention, FeedForward

# --- Configuration for a Fast, Toy Example ---
# We'll use much smaller values than in model.py to make this run instantly.
batch_size = 8
block_size = 16 # Max context length
max_iters = 20 # Just a few iterations to see the loss decrease
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 32 # Small embedding size
n_head = 4
n_layer = 2
dropout = 0.1
# --------------------------------------------

# Toy dataset
text = "hello world, this is a test to see how the model trains."
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

# Data loader
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Model & Optimizer ---
# We need to override the global hyperparameters from model.py for our test
model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed,
                  block_size=block_size, vocab_size=vocab_size, dropout=dropout)

# We can't use the DecoderTransformer class directly since it has hardcoded hyperparameters.
# Let's create a simple wrapper that accepts our test-specific arguments.
class TestDecoderTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding_table = nn.Embedding(args['vocab_size'], args['n_embed'])
        self.position_embedding_table = nn.Embedding(args['block_size'], args['n_embed'])
        self.blocks = nn.Sequential(*[Block(args['n_embed'], args['n_head'], args['block_size'], args['dropout']) for _ in range(args['n_layer'])])
        self.ln_f = nn.LayerNorm(args['n_embed'])
        self.lm_head = nn.Linear(args['n_embed'], args['vocab_size'])
        self.block_size = args['block_size']

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = TestDecoderTransformer(model_args)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("--- Starting a quick training simulation ---")
for i in range(max_iters):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
      print(f"Iteration {i}/{max_iters}, Loss: {loss.item():.4f}")

print("\n--- Simulation finished ---")
print("Loss has decreased, showing the model is learning.")

# --- Generate some text ---
print("\n--- Generating text from the test model ---")
start_context = torch.tensor(encode("hello "), dtype=torch.long, device=device).unsqueeze(0)
generated_text = decode(m.generate(start_context, max_new_tokens=50)[0].tolist())
print(f"Generated text: {generated_text}")
