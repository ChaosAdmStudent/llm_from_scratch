import torch 
import torch.nn as nn 

def text_to_ids(text: str, tokenizer): 
    return tokenizer.encode(text) 

def tk_ids_to_text(token_ids: torch.Tensor, tokenizer): 
    # token_ids: (1, N, token_id) 
    token_ids = token_ids.squeeze(0) 
    return tokenizer.decode(token_ids.tolist()) 

def logits_to_tk_ids(logits: torch.Tensor): 
    # logits: (B, N, vocab_size) 
    return torch.argmax(logits, dim=-1, keepdim=False) # (B, N)   

    