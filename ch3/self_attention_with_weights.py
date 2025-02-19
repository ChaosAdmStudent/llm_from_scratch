'''
In this file, I build the famous attention mechanism used in original transformer paper: The scaled dot-product attention. 
'''

import torch 
import torch.nn as nn 

class MySelfAttention_V1(nn.Module): 

    def __init__(self, inp_dim ,proj_dim, qkv_bias=False): 
        super(MySelfAttention_V1, self).__init__()
        self.proj_dim = proj_dim
        self.inp_dim = inp_dim  
        self.W_q = nn.Parameter(torch.rand(inp_dim, proj_dim, requires_grad=True)) 
        self.W_k = nn.Parameter(torch.rand(inp_dim, proj_dim, requires_grad=True)) 
        self.W_v = nn.Parameter(torch.rand(inp_dim, proj_dim, requires_grad=True)) 

    def forward(self, inputs): 
        
        # Create vector projections of the input token embeddings 
        Q = inputs @ self.W_q 
        K = inputs @ self.W_k
        V = inputs @ self.W_v

        # Calculate attention scores 
        omega = Q @ K.T  # Intermediate attention scores 
        attention_scores = torch.softmax(omega/self.proj_dim ** 0.5, dim=-1) # Final attention scores 

        # Calculate context vectors 
        Z = attention_scores @ V 

        return Z 

class MySelfAttention_V2(nn.Module): 

    def __init__(self, inp_dim ,proj_dim, qkv_bias=False): 
        super(MySelfAttention_V2, self).__init__()
        self.proj_dim = proj_dim
        self.inp_dim = inp_dim  
        self.W_q = nn.Linear(inp_dim, proj_dim, bias=qkv_bias) 
        self.W_k = nn.Linear(inp_dim, proj_dim, bias=qkv_bias) 
        self.W_v = nn.Linear(inp_dim, proj_dim, bias=qkv_bias) 

    def forward(self, inputs): 
        
        # Create vector projections of the input token embeddings 
        Q = self.W_q(inputs) 
        K = self.W_k(inputs) 
        V = self.W_v(inputs) 

        # Calculate attention scores 
        omega = Q @ K.T  # Intermediate attention scores 
        attention_scores = torch.softmax(omega/self.proj_dim ** 0.5, dim=-1) # Final attention scores 

        # Calculate context vectors 
        Z = attention_scores @ V 

        return Z 

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

    torch.manual_seed(123)

    inp_dim = input_token_embeddings.shape[-1] 
    proj_dim = 2 # Projected dimension 

    self_attention_block = MySelfAttention_V1(inp_dim, proj_dim) 
    Z = self_attention_block(input_token_embeddings) 
    print('Final context vectors using V1: ', Z)

    self_attention_block = MySelfAttention_V2(inp_dim, proj_dim) 
    Z = self_attention_block(input_token_embeddings) 
    print('Final context vectors using V2: ', Z) 

    print('Input token embeddings shape: ', input_token_embeddings.shape)
    print('Final context vectors shape: ', Z.shape)  


    # Exercise 3.1 
    # Checking if V1 and V2 produce same result if weights from V2 are copied to V1  
    # After checking documentation, nn.Linear performs x@W.T + b. Hence, W must have dimensions (out_features, in_features) 
    # But in nn.Parameter, weights has dimensions (in_features, out_features). Hence, I will transpose the weights before copying. 

    my_SA_V1 = MySelfAttention_V1(inp_dim, proj_dim) 
    my_SA_V2 = MySelfAttention_V2(inp_dim, proj_dim) 

    my_SA_V1.W_q.data = my_SA_V2.W_q.weight.data.T 
    my_SA_V1.W_k.data = my_SA_V2.W_k.weight.data.T
    my_SA_V1.W_v.data = my_SA_V2.W_v.weight.data.T 

    Z_V1 = my_SA_V1(input_token_embeddings)
    Z_V2 = my_SA_V2(input_token_embeddings)

    print('Final context vectors using V1: ', Z_V1)
    print('Final context vectors using V2: ', Z_V2)    
