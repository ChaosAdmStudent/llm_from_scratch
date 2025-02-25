import torch 
import tiktoken
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from ch4.gpt_model import GPTModel

def text_to_ids(text: str, tokenizer): 
    return tokenizer.encode(text) 

def tk_ids_to_text(token_ids: torch.Tensor, tokenizer): 
    # token_ids: (1, N, token_id) 
    token_ids = token_ids.squeeze(0) 
    return tokenizer.decode(token_ids.tolist()) 

def logits_to_tk_ids(logits: torch.Tensor): 
    # logits: (B, N, vocab_size) 
    return torch.argmax(logits, dim=-1, keepdim=False) # (B, N)   

def print_sampled_tokens(probas: torch.Tensor, vocab): 
    # probas ~ (B,N,vocab_size) 
    probas = probas.flatten(0,1) # (B*N, vocab_size) 
    sampled_token_ids = torch.tensor([torch.multinomial(probas, 1) for _ in range(1000)])
    sampled_bincount = torch.bincount(sampled_token_ids) 

    for token_id, freq in enumerate(sampled_bincount): 
        print(f'\t{vocab[token_id]} -> {freq} times') 

def generate(max_new_tokens: int, model, input_token_embeddings: torch.Tensor, context_length: int, device, temperature=1.0, top_k=None, eos_id=None): 
    
    """
    Takes input_token_embeddings and generates new tokens from the LLM model
    """ 
    model = model.to(device) 
    input_token_embeddings = input_token_embeddings.to(device) # (B, token_id) 

    for _ in range(max_new_tokens): 
        logits = model(input_token_embeddings[:, -context_length:]) # (B,N,vocab_size) 
        last_token_logits = logits[:, -1, :]  

        if top_k is not None: 
            top_logits, top_pos = torch.topk(last_token_logits, k=top_k, dim=-1) 
            last_token_logits = torch.where(
                condition = last_token_logits < top_logits[-1], # top_logits[-1] is the smallest element in top-k 
                input= torch.tensor(-torch.inf) , 
                other = last_token_logits # Retain values in indices where condition is False 
            ) 
        
        if temperature > 1: 
            last_token_logits = last_token_logits/temperature 
            next_token_probas = torch.softmax(last_token_logits,dim=-1) # (B, 1, vocab_size)   
            next_token_probas = next_token_probas.squeeze(1) # (B, vocab_size) 
            next_token_ids = torch.multinomial(next_token_probas, num_samples=1)  # (B,token_id) 
        else: 
            next_token_ids = torch.argmax(last_token_logits, dim=-1, keepdim=False) # (B,token_id) 
        
        if next_token_ids == eos_id: 
            break 
        
        # Merge token ids 
        input_token_embeddings = torch.cat((input_token_embeddings, next_token_ids), dim=0) # (B, token_id+1) 
    
    out_token_embeddings = input_token_embeddings
    return out_token_embeddings

     
if __name__ == '__main__': 
    pass 
