'''
In this file, I write code for evaluation of model  
'''

import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
import tiktoken 
from ch4.gpt_model import GPTModel
from ch5.utils import *

def cross_entropy_loss(outputs: torch.Tensor, target_tokens: torch.Tensor): 
    """
    outputs: Unnormalized logits output from GPT model (B, N, vocab_size) for corresponding target token ids
    targets: Target tokens for each next word (B, N) # We are predicting the same number of tokens in output as input (all shifted + next word) 
    """ 
    B,N = target_tokens.shape 
    token_indices = torch.arange(N)[None, :] 
    batch_indices = torch.arange(B)[:, None] 

    out_probas = torch.softmax(outputs, dim=-1) 
    out_probas = out_probas[batch_indices, token_indices, target_tokens]  # (B,N)
    
    # Calculate cross entropy using probabilities 
    losses = -torch.log(out_probas).mean(dim=-1)  # (B,1) 
    return losses 
    

if __name__ == '__main__': 
    tokenizer = tiktoken.get_encoding('gpt2') 
    GPT_CONFIG_124M = {
        "token_emb_dim": 768, 
        "droprate": 0.1, 
        "vocab_size": 50257, 
        "context_length": 256, 
        "num_heads": 12, 
        "num_layers": 12, 
        "qkv_bias": False 
    }    

    torch.manual_seed(123) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPTModel(GPT_CONFIG_124M) 
    model = model.to(device) 

    # Prepare text
    text = 'Every world tries to' 
    target_text = 'world tries to learn'

    inp_token_ids = text_to_ids(text, tokenizer) 
    inp_token_ids = torch.tensor(inp_token_ids, device=device).unsqueeze(0) # Add batch dimension 

    target_token_ids = text_to_ids(target_text, tokenizer) 
    target_token_ids = torch.tensor(target_token_ids, device=device).unsqueeze(0) 
    
    # Pass through GPT Model 
    logits = model(inp_token_ids) 

    my_ce_loss = cross_entropy_loss(logits, target_token_ids) 
    pytorch_ce_loss = nn.functional.cross_entropy(logits.flatten(0,1), target_token_ids.squeeze(0)) 

    print('My own cross entropy computed: ', my_ce_loss) 
    print('PyTorch cross entropy computed: ', pytorch_ce_loss) 

    perplexity = torch.exp(pytorch_ce_loss)
    print('Perplexity of model: ', perplexity) 
    print(f'Hence, model is unsure about {perplexity} number of words from vocabulary to choose for the next word prediction')

