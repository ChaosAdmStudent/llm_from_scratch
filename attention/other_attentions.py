'''
In this file, I implement other famous attention mechanisms, including Flash attention, multi-query attention and shared attention. 
''' 

import torch
import torch.nn as nn 
import math
from time import perf_counter

class MultiQueryAttention(nn.Module): 
    """
    Multiquery attention is like multihead attention but different heads share same set of keys and values
    """
    def __init__(self, token_emb_dim, context_dim, context_length, num_heads, droprate=0.1, qkv_bias=False): 
        super(MultiQueryAttention, self).__init__() 
        assert context_dim % num_heads == 0, "Context_dim should be divisible by num_heads!" 

        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = context_dim//num_heads 
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.float32), diagonal=1) 
        )

        self.W_query = nn.Linear(token_emb_dim, context_dim, bias=qkv_bias) 
        self.W_key = nn.Linear(token_emb_dim, self.head_dim, bias=qkv_bias) 
        self.W_value = nn.Linear(token_emb_dim, self.head_dim, bias=qkv_bias)  

        self.proj_out = nn.Linear(context_dim, context_dim) 
        self.dropout = nn.Dropout(droprate)

    def forward(self, input_emb: torch.Tensor): 
        assert len(input_emb.shape) == 3, "input_emb should be of shape (B, N, token_emb_dim)"
        B, num_tokens, inp_emb_dim = input_emb.shape
        
        # Create merged Q,K,V   
        Q = self.W_query(input_emb).contiguous() # (B, N, context_dim) 
        K = self.W_key(input_emb).contiguous() # (B, N, head_dim) 
        V = self.W_value(input_emb).contiguous() # (B, N, head_dim) 

        # Split the merged context_dim into different heads for query 
        Q = Q.view(B, num_tokens, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, N, head_dim)

        # Expand K and V to different num_heads, while sharing the same weights 
        K = K.unsqueeze(1) # (B,1,N,head_dim) 
        K = K.expand(B, self.num_heads, num_tokens, self.head_dim)  
        V = V.unsqueeze(1) 
        V = V.expand(B, self.num_heads, num_tokens, self.head_dim)  
        
        # Calculate attention scores 
        attention_scores = Q @ K.transpose(2,3) # (B,num_heads, N,N) 
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attention_weights = torch.softmax(attention_scores/math.sqrt(self.head_dim), dim=-1) 
        attention_weights = self.dropout(attention_weights) 

        # Calculate context vector 
        Z = attention_weights @ V # (B, num_heads, N, head_dim) 
        Z = Z.transpose(1,2).contiguous().view(B, num_tokens, self.context_dim) # (B, N, context_dim) 
        context_vec = self.proj_out(Z) 

        return context_vec  

