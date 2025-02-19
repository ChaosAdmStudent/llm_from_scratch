'''
In this file, I create sinusoidal absolute positional embeddings and add it to token embeddings to create the input embedding 
'''

import torch 
import torch.nn as nn 
from ch2.sliding_window import create_dataloader 
import tiktoken 
import math 

class SinusoidalEmbeddingLayer(): 

    def __init__(self, hidden_dim, context_length):
        assert hidden_dim % 2 == 0, 'Hidden dimension should be even to create sinusoidal embeddings' 
        self.hidden_dim = hidden_dim 
        self.context_length = context_length 

    def get_sinusoidal_embedding(self, positions): 
        pos_embeddings = torch.zeros(self.context_length, self.hidden_dim) 
        div_term = torch.exp(-torch.log(torch.tensor(10000)).float() * torch.arange(0,  self.hidden_dim, 2).float() / self.hidden_dim) 
        pos_embeddings[:, 0::2] = torch.sin(positions * div_term) 
        pos_embeddings[:, 1::2] = torch.cos(positions * div_term) 
        
        return pos_embeddings 

if __name__ == '__main__': 
    vocab_size = 50257 # GPT2 vocab size in BPE tokenizer 
    hidden_dim = 256 # Dimensionality of embedding vector 

    # Creating token embeddings 
    torch.manual_seed(123) 
    token_embedding_layer = nn.Embedding(vocab_size, hidden_dim) 

    with open('ch2/the-verdict.txt','r') as book: 
        raw_text = book.read() 

    max_length = 4 
    batch_size = 8 
    
    bpe = tiktoken.get_encoding('gpt2') 

    dataloader = create_dataloader(raw_text, bpe, max_length, stride=max_length, batch_size=batch_size, shuffle=False) 
    data_iter = iter(dataloader) 
    inputs, targets = next(data_iter) 
    
    print('Input shape:', inputs.shape)   
    token_embeddings = token_embedding_layer(inputs) 
    print('Token embedding shape: ',token_embeddings.shape) 
    
    # Creating positional embeddings 
    context_length = max_length  
    pos_embedding_layer = SinusoidalEmbeddingLayer(hidden_dim, context_length) 
    pos_embeddings = pos_embedding_layer.get_sinusoidal_embedding(torch.arange(context_length).unsqueeze(1)) # Unsqueezed to be able to do broadcasting 
    print(pos_embeddings.shape) 