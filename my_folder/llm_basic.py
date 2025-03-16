# -*- coding: utf-8 -*-
"""LLM basic

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PpNdDf4h7BK5Ww_l8FhxkF2xxX5UCxTu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tokenizer:

    @staticmethod
    def create_vocab(dataset):
        """
        Create a vocabulary from a dataset.

        Args:
            dataset (str): Text dataset to be used to create the character vocab.

        Returns:
            Dict[str, int]: Character vocabulary.
        """
        vocab = {
            token: index
            for index, token in enumerate(sorted(set(dataset)))
        }

        # Print tokens for debugging if needed
        for token in sorted(set(dataset)):
            print(token)

        # Adding unknown token
        vocab["<unk>"] = len(vocab)

        return vocab

    def __init__(self, vocab):
        """
        Initialize the tokenizer.

        Args:
            vocab (Dict[str, int]): Vocabulary.
        """
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        self.vocab_decode = {v: k for k, v in self.vocab_encode.items()}

    def encode(self, text):
        """
        Encode a text in level character.

        Args:
            text (str): Input text to be encoded.

        Returns:
            List[int]: List with token indices.
        """
        return [self.vocab_encode.get(char, self.vocab_encode["<unk>"]) for char in text]

    def decode(self, indices):
        """
        Decode a list of token indices.

        Args:
            indices (List[int]): List of token indices.

        Returns:
            str: The decoded text.
        """
        return "".join([self.vocab_decode.get(idx, "<unk>") for idx in indices])

# setting fix seed
torch.random.manual_seed(seed=1234)

# data
text = "Hi! My name is Matheus."
tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13] # ["Hi", "!", " My", " name", " is", " Mat", "he", "us", "."]

# parameters
vocab_size = max(tokens) + 1 # number of classes to predict
emb_dim = 5 # size of vector representation of each token
context = len(tokens) # context size of model

# layers
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
query = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
key = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
value = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)

# mask filter
ones = torch.ones(size=[context, context], dtype=torch.float)
mask = torch.tril(input=ones)

# forward pass
t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0) # [9] -> [1,9]
x = embedding(t_tokens) # [1,9] -> [1,9,50] embedding vectors

B, T, C = x.size()
Q = query(x) # [1,9,50] -> [1,9,50]
K = key(x) # [1,9,50] -> [1,9,50]
V = value(x) # [1,9,50] -> [1,9,50]

QK = Q @ K.transpose(-2, -1) * C**-0.5 # [1,9,50] @ [1,50,9] -> [1,9,9] attention matrix
attention = QK.masked_fill(mask[:T,:T] == 0, float("-inf")) # applying mask
attention = F.softmax(input=attention, dim=-1) # [1,9,9] normalizing to 0 and 1 in embedding dimension

out = attention @ V # [1,9,9] @ [1,9,50] -> [1,9,50]

print(out.size()) # new data representation

# setting fix seed
torch.random.manual_seed(seed=1234)

# data
# split : ["Hi", "!", " My", " name", " is", " Matheus", "."]
text = "Hi! My name is Matheus."
tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13]

# parameters
vocab_size = max(tokens) + 1 # number of classes to predict
emb_dim = 5 # size of vector representation of each token
context = len(tokens) # context size of model

# layers
pe = nn.Embedding(num_embeddings=context, embedding_dim=emb_dim) # learnable positional encoding
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
query = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
key = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
value = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)

# mask filter
ones = torch.ones(size=[context, context], dtype=torch.float)
mask = torch.tril(input=ones)

# token indices
indices = torch.arange(context, dtype=torch.long)

# forward pass
t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0) # [9] -> [1,9]
x = embedding(t_tokens) # [1,9] -> [1,9,50] embedding vectors
x = pe(indices) + x # [1,9,50] + [1,9,50] -> [1,9,50] positional encoding

B, T, C = x.size()
Q = query(x) # [1,9,50] -> [1,9,50]
K = key(x) # [1,9,50] -> [1,9,50]
V = value(x) # [1,9,50] -> [1,9,50]

QK = Q @ K.transpose(-2, -1) * C**-0.5 # [1,9,50] @ [1,50,9] -> [1,9,9] attention matrix
attention = QK.masked_fill(mask[:T,:T] == 0, float("-inf")) # applying mask
attention = F.softmax(input=attention, dim=-1) # [1,9,9] normalizing to 0 and 1 in column dimension

out = attention @ V # [1,9,9] @ [1,9,50] -> [1,9,50]

print(out.size()) # new data representation

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the Embedding layer with Positional Encoding.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pe = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the Embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        word_emb = self.embedding(x)
        word_pe = self.pe(x)
        return word_emb + word_pe

class AttentionBlock(nn.Module):

    def __init__(self, embedding_dim, context_size):
        """
        Initialize the AttentionBlock layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            context_size (int): Size of the context window.
        """
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        ones = torch.ones(size=[context_size, context_size], dtype=torch.float)
        self.register_buffer(name="mask", tensor=torch.tril(input=ones)) # Triangular matrix

    def forward(self, x):
        """
        Forward pass of the AttentionBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: New embedding representation of shape (batch_size, seq_len, embedding_dim).
        """
        B, T, C = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        qk = query @ key.transpose(-2, -1) * C**-0.5
        attention = qk.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        attention = F.softmax(input=attention, dim=-1)

        out = attention @ value
        return out

class MultiAttentionBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_size):
        """
        Initialize the MultiAttentionBlock layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            num_heads (int): Number of attention heads.
            context_size (int): Size of the context window.
        """
        super().__init__()

        # Checking number of heads
        head_dim = embedding_dim // num_heads
        assert head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.attention = nn.ModuleList(modules=[AttentionBlock(embedding_dim, head_dim, context_size) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x):
        """
        Forward pass of the MultiAttentionBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: New embedding representation of shape (batch_size, seq_len, embedding_dim).
        """
        out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
        x = self.linear(x)
        return x

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, ff_dim):
        """
        Initialize the feed forward layer.

        Args:
            emb_dim (int) : The dimension of the embedding.
            ff_dim (int) : The dimension of the feed forward layer.
            dropout_rate (float) : The dropout rate. (default: 0.2)
        """
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_dim, embedding_dim)


    def forward(self, x):
        """
        Forward pass of the feed forward layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, embedding_dim, head_dim, context_size, ff_dim):
        """
        Initialize the decoder layer.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            head_dim (int): Dimensionality of each head.
            context_size (int): Size of the context window.
            ff_dim (int): Dimensionality of the feed-forward layer.
        """
        super().__init__()
        self.attention = MultiAttentionBlock(embedding_dim, head_dim, context_size)
        self.feed_forward = FeedForward(embedding_dim, ff_dim)
        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x):
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        x_norm = self.norm_1(x)
        attention = self.attention(x_norm)
        attention = attention + x

        attention_norm = self.norm_2(attention)
        ff = self.feed_forward(attention_norm)
        ff = ff + attention

        return ff

# setting fix seed
torch.random.manual_seed(seed=1234)

# data
text = "Hi! My name is Matheus."
tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13] # ["Hi", "!", " My", " name", " is", " Mat", "he", "us", "."]

# parameters
vocab_size = max(tokens) + 1 # number of classes to predict
emb_dim = 5 # size of vector representation of each token
context = len(tokens) # context size of model

# layers
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
query = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
key = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)
value = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=False)

# mask filter
ones = torch.ones(size=[context, context], dtype=torch.float)
mask = torch.tril(input=ones)

# forward pass
t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0) # [9] -> [1,9]
x = embedding(t_tokens) # [1,9] -> [1,9,50] embedding vectors

B, T, C = x.size()
Q = query(x) # [1,9,50] -> [1,9,50]
K = key(x) # [1,9,50] -> [1,9,50]
V = value(x) # [1,9,50] -> [1,9,50]

QK = Q @ K.transpose(-2, -1) * C**-0.5 # [1,9,50] @ [1,50,9] -> [1,9,9] attention matrix
attention = QK.masked_fill(mask[:T,:T] == 0, float("-inf")) # applying mask
attention = F.softmax(input=attention, dim=-1) # [1,9,9] normalizing to 0 and 1 in embedding dimension

out = attention @ V # [1,9,9] @ [1,9,50] -> [1,9,50]
print(attention)
print(V)
print(Q)
print(out)
print(out.size()) # new data representation





