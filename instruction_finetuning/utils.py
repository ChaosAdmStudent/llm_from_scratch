import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretraining.utils import generate
from instruction_finetuning.data_prep import format_input_alpaca
from tqdm import tqdm 
import json 
import torch 

def generate_out_text_response(model, input_text, input_token_embedding, context_length, tokenizer, device) -> str: 
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
 
    model.toggle_kv_cache(False) 

    output_text = tokenizer.decode(out_tk_ids.tolist()) 
    return output_text[len(input_text):] 

def generate_model_responses(file_path: str, model, test_data, tokenizer, context_length, device): 
    model.eval() 
    model.to(device) 
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)): 
        input_text = format_input_alpaca(entry) 
        input_token_ids = torch.tensor([tokenizer.encode(input_text)], device=device)   
        response = generate_out_text_response(model, input_text, input_token_ids, context_length, tokenizer, device) 
        response = response.replace('### Response: ', '') 
        test_data[i]['model_response'] = response 
    
    with open(file_path, 'w') as file: 
        json.dump(test_data, file, indent=4) 