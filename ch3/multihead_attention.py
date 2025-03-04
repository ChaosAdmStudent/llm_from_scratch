'''
In this file, I build the multi-head attention mechanism, both inefficient and efficient versions. 

THEORY: 
In multi-head attention, we are creating a wrapper class that creates multiple instances or "heads" of the Causal Self Attention. 
The efficient version does computation of each head in parallel. 
''' 

import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import torch 
import torch.nn as nn 
from ch3.casual_self_attention import CausalSelfAttention
from dataclasses import dataclass

@dataclass
class ModelArgs: 

    # Required for KV Cache
    max_batch_size: int = 8, 
    max_seq_len: int = 2048

class MultiHeadAttentionWrapperClass_V1(nn.Module): 
    """
    Inefficient (Non-Parallel) version of multi-head attention. 
    This involves creating multiple instances of the CausalSelfAttention and then using the Wrapper class to call each instance one by one. 
    """

    def __init__(self, inp_emb_dim, proj_dim, context_length, dropout, qkv_bias=False, num_heads=4):  
        super(MultiHeadAttentionWrapperClass_V1, self).__init__() 
        self.num_heads = num_heads 
        self.heads = nn.ModuleList([
            CausalSelfAttention(inp_emb_dim,proj_dim, context_length, dropout, qkv_bias=qkv_bias) for _ in range(num_heads) 
        ]) 

    def forward(self, x): 
        return torch.cat([head(x) for head in self.heads], dim=-1) # dim = -1 concatenates the context vectors along the projected dimension 
        # Thus if there were 2 heads and input token embedding was shape (2,6,3), then the output will be (2,6,6) 

class MultiHeadAttention_V2(nn.Module): 
    """
    Efficient implementation of multi-head attention. 
    This is the formal way of doing it. We don't use a wrapper class that calls many instances of CausalSelfAttention. 
    Instead, we integrate the operations of multiple heads into a single Self Attention class. 
    We leverage matrix multiplications instead of heavy for loops to do this. 
    Essentially, we are splitting the weight matrices of Q, K and V into multiple heads. 
    """

    def __init__(self, inp_emb_dim, context_dim, context_length, dropout,num_heads=4,qkv_bias=False):  
        super(MultiHeadAttention_V2, self).__init__() 
        
        assert context_dim % num_heads == 0, "Required context_dim should be divisible by num_heads"
        
        self.inp_emb_dim = inp_emb_dim 
        self.context_length = context_length
        self.context_dim = context_dim
        self.num_heads = num_heads  
        self.head_dim = context_dim // num_heads 
        self.W_query = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.W_key = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.W_value = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.kv_cache_enabled = False  

        self.dropout = nn.Dropout(dropout) 
        self.proj_out = nn.Linear(context_dim, context_dim) # Used to merge the different heads. Optional but used in LLM architectures heavily. 
        self.register_buffer( # Mask for causal self attention
            'mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1) 
        )

        self.register_buffer("cache_k", None) # Buffers for caches 
        self.register_buffer("cache_v", None)  

    def forward(self, inputs, args:ModelArgs = None, start_pos: int = None): 
        assert len(inputs.shape) == 3, "Input must be of shape (num_batches, num_tokens, token_dimensions)"  
        assert inputs.shape[-1] == self.inp_emb_dim, "Input hidden dimension must be equal to inp_emb_dim passed into MHA!"
        if self.kv_cache: 
            assert start_pos is not None, "Must provide start_pos argument if using kv-cache"

        B, num_tokens, _ = inputs.shape 
        
        # Create merged K,Q,V 
        K = self.W_key(inputs) # Shape = (B,N,context_dim) 
        Q = self.W_query(inputs) 
        V = self.W_value(inputs) 

        # Split K,Q,V into multiple heads 

        # Split context_dim into different heads 
        K = K.view(B, num_tokens, self.num_heads, self.head_dim) # Splits the last context_dim across heads. 
        Q = Q.view(B, num_tokens, self.num_heads, self.head_dim) 
        V = V.view(B, num_tokens, self.num_heads, self.head_dim) 

        if self.kv_cache_enabled: 
            if self.cache_k is None: 
                # Initialize cache for keys and values 
                self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.num_heads, self.head_dim)
                self.cache_v = torch.zeros_like(self.cache_k) 

            # Cache keys and values 
            self.cache_k[:B, start_pos: start_pos + num_tokens] = K 
            self.cache_v[:B, start_pos: start_pos + num_tokens] = V 

            # Extract cached keys and values for computations 
            K = self.cache_k[:B, :start_pos + num_tokens] # (B, N', num_heads, head_dim) 
            V = self.cache_v[:B, :start_pos + num_tokens] 

        # Switch positions of num_heads so that batch matrix multiplication can be done across several heads in parallel 
        K = K.transpose(1,2) # (B,num_heads, N, head_dim) OR  (B, num_heads, N' , head_dim) if kv_Cache
        Q = Q.transpose(1,2) # (B,num_heads, N, head_dim) OR (B, num_heads, 1, head_dim) if kv_cache
        V = V.transpose(1,2)

        # Calculate attention weights 
        attention_scores = Q @ K.transpose(2,3) # (B, num_heads , N, N) or (B, num_heads, 1, N') if kv_cache 
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Causal self attention
        attention_weights = torch.softmax(attention_scores/K.shape[-1] ** 0.5, dim=-1) # d_k is the dimension per head in this case

        # Dropout on attention weights 
        attention_weights = self.dropout(attention_weights)

        # Calculate context vectors 
        Z = attention_weights @ V # (B, num_heads, N, head_dim) 
        Z = Z.transpose(1,2).contiguous() # (B, N, num_heads, head_dim)  
        Z = Z.view(B, num_tokens, self.context_dim)  # (B, N, context_dim) 

        # Linearly project merged head information to get final context vector 
        context_vec = self.proj_out(Z) # (B, N, context_dim) 
        return context_vec 

if __name__ == '__main__': 
    # Simulate generated input embeddings 
    input_token_embeddings = torch.tensor([
        [0.43, 0.15, 0.89], # Your  
        [0.55, 0.87, 0.66], # Journey 
        [0.57, 0.85, 0.64], # Starts 
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55] # step 
    ])  

    #  Simulate batched input embeddings 
    input_token_embeddings = torch.stack((input_token_embeddings, input_token_embeddings)) 

    # Exercise 3.2 (We need output context vector to have 2 dimensions) 
    # Exercise super simple. Just use proj_dim=1 and num_heads=2 
    inp_token_dim = input_token_embeddings.shape[-1] 
    proj_dim = 1   
    dropout = 0.5 
    context_length = input_token_embeddings.shape[1] 
    num_heads = 2 

    single_head_attention = CausalSelfAttention(inp_token_dim, proj_dim, context_length, dropout, qkv_bias=False)  
    multi_head_attention_v1 = MultiHeadAttentionWrapperClass_V1(inp_token_dim, proj_dim, context_length, dropout, num_heads=num_heads) 
    
    Z_single_head = single_head_attention(input_token_embeddings) 
    Z_multi_head_v1 = multi_head_attention_v1(input_token_embeddings) 

    print('Input token embedding shape: ', input_token_embeddings.shape) 
    print('Number of heads: ', num_heads)
    print('Single head attention output shape: ', Z_single_head.shape) 
    print('Multi head attention V1 output shape: ', Z_multi_head_v1.shape) 

    ###################################

    # Exercise 3.3 (GPT2 module parameters)

    context_length = 1024 # Num_tokens
    inp_token_dim = 768 # Input token dimension
    context_dim = inp_token_dim # Context vector projected dimension  

    input_token_embeddings = torch.randn(2, context_length, inp_token_dim) 
    print('Input token embedding shape: ', input_token_embeddings.shape) 

    num_heads = 12
    single_head_attention = CausalSelfAttention(inp_token_dim, context_dim, context_length, dropout, qkv_bias=False) 
    multi_head_attention_v2 = MultiHeadAttention_V2(inp_token_dim, context_dim, context_length, dropout, num_heads=num_heads) 

    Z_single_head = single_head_attention(input_token_embeddings)
    Z_multi_head_v2 = multi_head_attention_v2(input_token_embeddings) 

    print('Number of heads: ', num_heads)
    print('Single head attention output shape: ', Z_single_head.shape) 
    print('Multi head attention V2 output shape: ', Z_multi_head_v2.shape) 

    #####################################







