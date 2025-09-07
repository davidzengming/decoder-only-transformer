import torch
import torch.nn as nn
from torch.nn import functional as F

# We are re-defining the classes here from model.py so we can inject
# print statements without modifying our original, clean code.

# --- Hyperparameters for this debug run ---
block_size = 16
n_embed = 32
n_head = 4
n_layer = 2
dropout = 0.1
device = 'cpu' # Force CPU for simplicity in debugging
# --------------------------------------------

# --- Re-defined Model Components with Debug Prints ---

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.n_head = n_head
        self.n_embed = n_embed
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels (n_embed)
        print(f"    [ATTN] Input to Attention Head:      shape={x.shape}")

        # --- Q, K, V Calculation ---
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        print(f"    [ATTN] After splitting into Q, K, V:  q={q.shape}, k={k.shape}, v={v.shape}")

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        print(f"    [ATTN] After reshaping for heads:    q={q.shape} (Batch, n_heads, Time, head_size)")

        # --- Attention Score Calculation ---
        att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
        print(f"    [ATTN] Attention scores (affinities): shape={att.shape} (How much each token attends to others)")

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        print(f"    [ATTN] After applying causal mask (hiding future tokens)")
        
        att = F.softmax(att, dim=-1)
        print(f"    [ATTN] After softmax (scores become probabilities)")

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        print(f"    [ATTN] Output after multiplying by V: shape={y.shape} (Weighted sum of values)")
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        print(f"    [ATTN] Final output of Attention Head:  shape={y.shape}")
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        print(f"\n  [BLOCK] Entering Transformer Block")
        print(f"  [BLOCK] Input shape: {x.shape}")
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        print(f"  [BLOCK] Output shape: {x.shape}")
        return x

# --- The Walkthrough ---

# 1. Input Text and Vocabulary
text = "hello"
chars = sorted(list(set("abcdefghijklmnopqrstuvwxyz ,."))) # A bigger vocab
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
input_ids = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)

print("--- STEP 1: INPUT & EMBEDDINGS ---")
print(f"Original text: '{text}'")
print(f"Input tensor (token IDs): {input_ids}, shape={input_ids.shape}")

# 2. Model Initialization
token_embedding_table = nn.Embedding(vocab_size, n_embed)
position_embedding_table = nn.Embedding(block_size, n_embed)
tok_emb = token_embedding_table(input_ids)
pos_emb = position_embedding_table(torch.arange(input_ids.shape[1], device=device))
x = tok_emb + pos_emb

print(f"Token embeddings shape:   {tok_emb.shape}")
print(f"Position embeddings shape: {pos_emb.shape}")
print(f"Final input to blocks (tok_emb + pos_emb): shape={x.shape}")

# 3. Transformer Blocks
blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
y = blocks(x)

# 4. Final Output
ln_f = nn.LayerNorm(n_embed)
lm_head = nn.Linear(n_embed, vocab_size)
y = lm_head(ln_f(y))

print("\n--- STEP 2: FINAL OUTPUT ---")
print(f"Output from blocks before final layer: shape={x.shape}")
print(f"Final output (logits) shape: {y.shape} (Batch, Time, Vocab_Size)")
print("\nEach of the 5 tokens in the sequence now has a vector of 29 logits,")
print("predicting the probability of the next token in the vocabulary.")
