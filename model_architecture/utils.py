import torch 
import torch.nn as nn 

class GELU(nn.Module): 
    def __init__(self): 
        super(GELU, self).__init__() 

    def forward(self, x): 
        return (0.5 * x * (1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x,3)))
        )
        ) 

def generate_new_tokens(model, max_new_tokens: int, token_ids: torch.Tensor, context_length: int, use_kv_cache=False): 
    """
    max_new_tokens: Number of tokens you want to generate 
    token_ids: Input tensor with token IDs. Needs to have shape (B, num_tokens) 
    context_length: The max context supported by LLM model. If it's less than num_tokens, then I pick context_length number of tokens from the end 
    """ 

    B, num_tokens = token_ids.shape 

    if use_kv_cache: 
        pass 

    else: 
        for _ in range(max_new_tokens): 
            with torch.no_grad(): 
                logits = model(token_ids[:, -context_length:])  # (B, num_tokens, vocab_size) 
            logits = logits[:, -1, :] # Extracting logits for the last token. That's the new word we are interested in
            probs = torch.softmax(logits, dim=-1)   
            new_token_id = torch.argmax(probs, dim=-1, keepdim=True) 
            token_ids = torch.cat((token_ids, new_token_id), dim=1) # Adding token in the last dimension. Thus, we get (B, num_tokens+1) 
        
        return token_ids 


