'''
In this file, I create Learned absolute positional embeddings and add it to token embeddings to create the input embedding 
'''

import torch 
import torch.nn as nn 
from sliding_window import create_dataloader 
import tiktoken 

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
    pos_embedding_layer = nn.Embedding(context_length, hidden_dim) 
    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) 

    print('Positional embeddings shape:', pos_embeddings.shape)  

    # Adding positional embeddings to token embeddings to get input embeddings 
    input_embeddings = token_embeddings + pos_embeddings 
    print('Input embeddings shape:', input_embeddings.shape) 