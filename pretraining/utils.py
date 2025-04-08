import torch 
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from attention.multihead_attention import ModelArgs

def text_to_ids(text: str, tokenizer): 
    return torch.tensor([tokenizer.encode(text)])   

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

def generate(max_new_tokens: int, model, input_token_embeddings: torch.Tensor, context_length: int, device, temperature=0.0, top_k=None, eos_id=50256, use_kv_cache=False): 
    
    """
    Takes input_token_embeddings and generates new tokens from the LLM model 
    """ 

    if use_kv_cache: 
        args = ModelArgs() 
        assert len(input_token_embeddings) <= args.max_batch_size, f"Input should not have more than {args.max_batch_size} batches" 
        assert context_length <= args.max_seq_len, f"Input context length should not have more than {args.max_seq_len} sequence length"

        total_len = min(args.max_seq_len, min(context_length, input_token_embeddings.shape[1]) + max_new_tokens - 1) # This is the total length of the generated sequence since we return what was input as well.
    else: 
        total_len = max_new_tokens 
    
    model = model.to(device) 
    input_token_embeddings = input_token_embeddings.to(device) # (B, token_id) 
    out_token_embeddings = input_token_embeddings # Initialize output token embedding sequence with the input (This will be extended with new tokens in this function)

    cur_pos = 0
    
    with torch.no_grad(): 
        while cur_pos < total_len:
            if cur_pos == 0 and use_kv_cache: 
                logits = model(input_token_embeddings[:, -context_length:], start_pos=0) # (B,N,vocab_size) 
                last_token_logits = logits[:, -1, :]   # (B, vocab_size)
                cur_pos += input_token_embeddings.shape[1] - 1 # These number of tokens have been seen for each batch (Current assumption: All batches have same number of tokens) 

            elif use_kv_cache: 
                last_token_logits = model(next_token_ids, start_pos=cur_pos) # (B, 1, vocab_size) 
                last_token_logits = last_token_logits.squeeze(1) # (B, vocab_size) 

            else: 
                logits = model(out_token_embeddings[:, -context_length:]) # (B,N,vocab_size) 
                last_token_logits = logits[:, -1, :] # (B, vocab_size) 

            if top_k is not None: 
                top_logits, top_pos = torch.topk(last_token_logits, k=top_k, dim=-1) 
                last_token_logits = torch.where( 
                    condition = last_token_logits < top_logits[:, [-1]], # top_logits[-1] is the smallest element in top-k 
                    input= torch.full_like(last_token_logits, -torch.inf), 
                    other = last_token_logits # Retain values in indices where condition is False 
                ) 
            
            if temperature > 0: 
                last_token_logits = last_token_logits/temperature 
                next_token_probas = torch.softmax(last_token_logits,dim=-1) # (B, vocab_size)   
                next_token_ids = torch.multinomial(next_token_probas, num_samples=1)  # (B,1) 
            else: 
                next_token_ids = torch.argmax(last_token_logits, dim=-1, keepdim=True) # (B,1) 

            if next_token_ids.shape[0] == 1 and next_token_ids[0] == eos_id: # Only stop generation if batch size of 1 is given (inference)
                break 
            
            out_token_embeddings = torch.cat((out_token_embeddings, next_token_ids), dim=1) # (B, token_id+1) 
            cur_pos += 1 
    
    return out_token_embeddings 