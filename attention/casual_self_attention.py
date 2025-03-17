'''
In this file, I build the causal self-attention mechanism, typically used in the decoder block of the transformer model. 
It helps for LLM tasks like language modelling where next word prediction should only be able to look at previous words to predict the next one. 
'''

import torch 
import torch.nn as nn 

class CausalSelfAttention(nn.Module): 
    def __init__(self, inp_emb_dim, proj_dim, context_length, dropout, qkv_bias=False): 
        super(CausalSelfAttention, self).__init__() 
        self.inp_emb_dim = inp_emb_dim
        self.proj_dim = proj_dim
        self.W_q = nn.Linear(inp_emb_dim, proj_dim, bias=qkv_bias)
        self.W_k = nn.Linear(inp_emb_dim, proj_dim, bias=qkv_bias) 
        self.W_v = nn.Linear(inp_emb_dim, proj_dim, bias=qkv_bias) 
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1) # Diagonal = 1 masks out only ABOVE main diagonal 
        )
     
    def forward(self, inputs): 
        assert inputs.shape[-1] == self.inp_emb_dim, 'Input embeddings dimension should match the model input dimension'
        assert len(inputs.shape) == 3, 'Input should be of shape (batch_size, sequence_length, embedding_dim)' 

        b, num_tokens, inp_dim = inputs.shape  # Get the length of the input sequence

        # Create vector projections of the input token embeddings 
        Q = self.W_q(inputs) 
        K = self.W_k(inputs)
        V = self.W_v(inputs)
        
        attention_scores = Q @ torch.transpose(K,1,2) # Intermediate/Unnormalized attention scores 
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Apply causal masking. -inf will be masked out by softmax 
        attention_weights = torch.softmax(attention_scores/self.proj_dim ** 0.5, dim=-1) # Final normalized attention weights 
        attention_weights = self.dropout(attention_weights) # Apply dropout to attention weights to avoid overfitting 

        # Calculate context vectors 
        Z = attention_weights @ V

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

    #  Simulate batched input embeddings 
    input_token_embeddings = torch.stack((input_token_embeddings, input_token_embeddings))  
    print('Input token embeddings shape: ', input_token_embeddings.shape)

    torch.manual_seed(123) 

    inp_dim = 3 # Token embeddings dimension 
    proj_dim = 3 # Projected dimensions for Q,K,V. Typically same as inp_dim 
    context_length = input_token_embeddings.shape[1] # Length of the input sequence  
    dropout = 0.5 # Dropout probability 

    self_attention_block = CausalSelfAttention(inp_dim, proj_dim, context_length, dropout) 
    context_vectors = self_attention_block(input_token_embeddings) 

    print('Context vectors shape: ', context_vectors.shape) 
    print('Context vectors: \n', context_vectors)