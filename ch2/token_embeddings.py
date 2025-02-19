import torch
import torch.nn as nn 

if __name__ == '__main__': 
    vocab_size = 6  # Number of tokens 
    out_dim = 5  # Number of dimensions in the embedding 

    torch.manual_seed(123) 
    embeddings = nn.Embedding(vocab_size, out_dim) 
    print(embeddings.weight) 
    print(embeddings(torch.tensor([2,1])))


    