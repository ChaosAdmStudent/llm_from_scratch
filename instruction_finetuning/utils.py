import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretraining.utils import generate

def generate_out_text_response(model, input_text, input_token_embedding, context_length, tokenizer, device): 
    """
    Generates output responose for an input_token_embedding. Input has shape: (1, num_tokens) 
    """
    
    model.eval() 
    model.toggle_kv_cache(True) 
    
    out_tk_ids = generate(
        max_new_tokens=35, 
        model= model, 
        input_token_embeddings=input_token_embedding, 
        context_length=context_length, 
        device=device, 
        use_kv_cache=True
    )[0] 

    model.train() 
    model.toggle_kv_cache(False) 

    output_text = tokenizer.decode(out_tk_ids.tolist()) 
    return output_text[len(input_text):] 
