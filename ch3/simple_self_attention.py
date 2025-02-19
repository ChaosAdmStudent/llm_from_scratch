'''
In this script, I code a simple self attention mechanism without any trainable weights just to get the idea across. 
Self attention works by finding a context vector for every token embedding. This context vector has enriched context about the query token and helps modify its initial token
embedding to a more contextually enriched embedding. 
''' 

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn 
import tiktoken 
from ch2.sinusoidal_PE import SinusoidalEmbeddingLayer 

def self_attention_one_token(input_token_embeddings): 
    # Simple Self Attention Mechanism for one query token 
    # I will use the 2nd token as the query token 

    # Intermediate attention weight computation (omega) 
    query_token = input_token_embeddings[1]  
    print('Query token shape: ', query_token.shape)
    print('Query token: ', query_token)
    intermediate_attention_scores = torch.empty(len(input_token_embeddings)) 
    for i, input_embedding in enumerate(input_token_embeddings): 
        intermediate_attention_scores[i] = torch.dot(input_embedding, query_token) 
    
    print('Intermediate attention weights: ', intermediate_attention_scores) 

    # Softmax to normalize intermediate attention weights to get final attention weights. 
    attention_scores = torch.softmax(intermediate_attention_scores, dim=0) 

    print('Final attention weights: ', attention_scores) 
    print('Sum of attention weights: ', attention_scores.sum().item())

    # Context vector computation

    context_vector = torch.zeros(3) 
    for i, attention_score in enumerate(attention_scores): 
        context_vector += attention_score * input_token_embeddings[i] 

    print('Context vector shape: ', context_vector.shape)
    print('Context vector: ', context_vector)

def self_attention(inputs): # For all tokens 

    # Calculate attention scores 
    intermediate_attention_scores = inputs @ inputs.T 
    final_attention_scores = torch.softmax(intermediate_attention_scores, dim=-1) 
    context_vectors = final_attention_scores @ inputs 

    print('Final context vectors: ', context_vectors) 


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

    print('Input token embeddings shape: ', input_token_embeddings.shape)
    print('Input embeddings: \n', input_token_embeddings)

    self_attention(input_token_embeddings)
