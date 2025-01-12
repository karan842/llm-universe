import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple vanilla Transformer model with positional encoding
class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=512):
        super(VanillaTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, max_len)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def _generate_positional_encoding(self, d_model, max_len):
        # Using a standard sinusoidal positional encoding
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x: input sequence of token indices
        emb = self.embedding(x) + self.positional_encoding[:, :x.size(1)]
        emb = emb.permute(1, 0, 2)  # Required shape for transformer (seq_len, batch_size, d_model)
        output = self.transformer(emb, emb)  # Self-attention layer
        output = self.fc_out(output)
        return output

# Sample sentences and parameters
sentences = ["The cat sat on the mat.", "A dog ran fast."]
vocab = {"<PAD>": 0, "The": 1, "cat": 2, "sat": 3, "on": 4, "the": 5, "mat": 6, "A": 7, "dog": 8, "ran": 9, "fast": 10}
vocab_size = len(vocab)
d_model = 16  # Embedding dimension
nhead = 2
num_layers = 2

def tokenize(sentences, vocab, pad_token="<PAD>"):
    tokenized = []
    max_length = max(len(sentence.split()) for sentence in sentences)  # Find the longest sentence
    pad_idx = vocab[pad_token]  # Index for the <PAD> token

    for sentence in sentences:
        tokens = sentence.lower().split()  # Simple whitespace-based tokenization
        token_indices = [vocab.get(token, pad_idx) for token in tokens]  # Map tokens to indices
        # Pad the sentence to the max length
        tokenized.append(token_indices + [pad_idx] * (max_length - len(token_indices)))
    
    return torch.tensor(tokenized)

tokenized_sentences = tokenize(sentences, vocab)
model = VanillaTransformer(vocab_size, d_model, nhead, num_layers)

# Forward pass through the model
output = model(tokenized_sentences)
print(output.shape)
