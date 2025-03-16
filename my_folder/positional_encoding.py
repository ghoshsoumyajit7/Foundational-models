import torch
import torch.nn as nn
import torch.nn.functional as F

# setting fix seed
torch.random.manual_seed(seed=1234)

# data
# split : ["Hi", "!", " My", " name", " is", " Matheus", "."]
text = "Hi! My name is Matheus."
tokens = [13347, 0, 3092, 836, 374, 7011, 383, 355, 13]

# parameters
vocab_size = max(tokens) + 1  # number of classes to predict
emb_dim = 5  # size of vector representation of each token
context = len(tokens)  # context size of model

# layers
pe = nn.Embedding(num_embeddings=context, embedding_dim=emb_dim)  # learnable positional encoding
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
t_tokens = torch.tensor(data=tokens).unsqueeze(dim=0)  # [9] -> [1,9]
x = embedding(t_tokens)  # [1,9] -> [1,9,5] embedding vectors
x = pe(indices) + x  # [1,9,5] + [1,9,5] -> [1,9,5] positional encoding

B, T, C = x.size()
Q = query(x)  # [1,9,5] -> [1,9,5]
K = key(x)  # [1,9,5] -> [1,9,5]
V = value(x)  # [1,9,5] -> [1,9,5]

QK = Q @ K.transpose(-2, -1) * C**-0.5  # [1,9,5] @ [1,5,9] -> [1,9,9] attention matrix
attention = QK.masked_fill(mask[:T, :T] == 0, float("-inf"))  # applying mask
attention = F.softmax(input=attention, dim=-1)  # [1,9,9] normalizing to 0 and 1 in column dimension

out = attention @ V  # [1,9,9] @ [1,9,5] -> [1,9,5]

print(out.size())  # new data representation
